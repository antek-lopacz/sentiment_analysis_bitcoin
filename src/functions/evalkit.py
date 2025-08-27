from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

# =======================
# Evaluation utilities
# =======================

def _as_numpy(y) -> np.ndarray:
    try:
        return y.detach().cpu().numpy()
    except Exception:
        return np.asarray(y)

def _parse_batch(batch):
    """
    Returns (features, labels) where features is either a tensor or a dict.
    Supports:
      - (inputs, labels)
      - (input_ids, attention_mask, labels)
      - dict directly (caller then provides labels separately)
    """
    if isinstance(batch, tuple) or isinstance(batch, list):
        if len(batch) == 2:
            x, y = batch
            if isinstance(x, dict):
                return x, y
            return x, y
        elif len(batch) == 3:
            input_ids, attention_mask, labels = batch
            return {"input_ids": input_ids, "attention_mask": attention_mask}, labels
        else:
            raise ValueError("Unsupported batch structure.")
    elif isinstance(batch, dict):
        return batch, None
    else:
        raise ValueError("Unsupported batch type.")

def _forward(model, features, device):
    if isinstance(features, dict):
        ids = features["input_ids"].to(device, non_blocking=True)
        mask = features.get("attention_mask", None)
        mask = mask.to(device, non_blocking=True) if mask is not None else None
        return model(ids, mask)
    else:
        xb = features.to(device, non_blocking=True)
        return model(xb)

def evaluate(
    model: torch.nn.Module,
    X: Union[torch.utils.data.DataLoader, torch.Tensor, Dict[str, torch.Tensor]],
    y: Optional[torch.Tensor],
    dataset_name: str = "Dataset",
    *,
    device: Union[torch.device, str] = "cpu",
    batch_size: int = 128,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a model on a DataLoader or raw tensors/dicts.
    Returns a dict with accuracy, f1_macro, f1_weighted.
    """
    model.eval()
    preds_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    with torch.inference_mode():
        # Case 1: DataLoader
        if hasattr(X, "__iter__") and not isinstance(X, (torch.Tensor, dict)):
            for batch in X:
                feats, labels = _parse_batch(batch)
                logits = _forward(model, feats, device)
                preds = torch.argmax(logits, dim=1)
                preds_list.append(preds.cpu())
                if labels is None:
                    raise ValueError("Dataloader must yield labels.")
                labels_list.append(labels.cpu())

        # Case 2: dict of inputs (BERT-style) + tensor labels
        elif isinstance(X, dict):
            if y is None:
                raise ValueError("Labels must be provided when X is a dict.")
            n = X["input_ids"].size(0)
            for i in range(0, n, batch_size):
                chunk = {k: v[i:i+batch_size] for k, v in X.items()}
                logits = _forward(model, chunk, device)
                preds = torch.argmax(logits, dim=1)
                preds_list.append(preds.cpu())
                labels_list.append(y[i:i+batch_size].cpu())

        # Case 3: single tensor inputs (MLP-style) + tensor labels
        elif isinstance(X, torch.Tensor):
            if y is None:
                raise ValueError("Labels must be provided when X is a tensor.")
            n = X.size(0)
            for i in range(0, n, batch_size):
                xb = X[i:i+batch_size]
                logits = _forward(model, xb, device)
                preds = torch.argmax(logits, dim=1)
                preds_list.append(preds.cpu())
                labels_list.append(y[i:i+batch_size].cpu())
        else:
            raise ValueError("Unsupported X type for evaluate.")

    y_pred = torch.cat(preds_list).numpy()
    y_true = torch.cat(labels_list).numpy()

    if y_pred.shape[0] != y_true.shape[0]:
        raise RuntimeError(f"Pred/true length mismatch: {y_pred.shape[0]} vs {y_true.shape[0]}")

    acc = (y_pred == y_true).mean()
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    if verbose:
        print(f"{dataset_name} set:  Accuracy: {acc:.4f}  F1(macro): {f1_macro:.4f}  F1(weighted): {f1_weighted:.4f}")

    return {"accuracy": float(acc), "f1_macro": float(f1_macro), "f1_weighted": float(f1_weighted)}

# =======================
# Reporting helpers
# =======================

def _extract_metrics_from_report_df(df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    sklearn's classification_report saved as DataFrame has accuracy stored in an odd place.
    Try the common patterns and fall back safely.
    """
    if "accuracy" in df.index and "precision" in df.columns:
        acc = float(df.loc["accuracy", "precision"])
    elif "accuracy" in df.index:
        acc = float(df.loc["accuracy"].to_numpy()[0])
    else:
        # compute from per-class if needed (rare)
        acc = float(df.get("accuracy", pd.Series([np.nan])).fillna(method="ffill").iloc[0])
    macro_f1 = float(df.loc["macro avg", "f1-score"])
    weighted_f1 = float(df.loc["weighted avg", "f1-score"])
    return acc, macro_f1, weighted_f1

def generate_svm_reports(
    X_train: np.ndarray,
    y_train,
    X_test: np.ndarray,
    y_test,
    output_dir: Path,
    *,
    prefix: str = "SVM",
    estimator: Optional[Pipeline] = None,
    param_grid: Optional[Dict[str, Iterable[Any]]] = None,
    grid_search=None,
    labels: Optional[Sequence[int]] = None,
    digits: int = 4,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """
    For each hyperparameter combo, fit an estimator, write per-combo report CSVs,
    and save a summary CSV in output_dir.
    Accepts either a fitted GridSearchCV (to reuse its param list) or estimator+param_grid.
    """
    from joblib import Parallel, delayed

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if grid_search is not None:
        base_estimator = grid_search.estimator
        params_list = list(grid_search.cv_results_["params"])
    else:
        if estimator is None or param_grid is None:
            raise ValueError("Provide either grid_search OR (estimator AND param_grid).")
        base_estimator = estimator
        params_list = list(ParameterGrid(param_grid))

    y_train_np = _as_numpy(y_train)
    y_test_np = _as_numpy(y_test)

    def _worker(params: Dict[str, Any]) -> Tuple[str, float, float, float]:
        c = params.get("svc__C", "NA")
        g = params.get("svc__gamma", "NA")
        k = params.get("svc__kernel", "NA")
        fname = f"{prefix}_C_{c}_gamma_{g}_kernel_{k}.csv"
        fpath = output_dir / fname

        if fpath.exists():
            df = pd.read_csv(fpath, index_col=0)
            acc, macro_f1, weighted_f1 = _extract_metrics_from_report_df(df)
            return fname, acc, macro_f1, weighted_f1

        est = clone(base_estimator).set_params(**params)
        est.fit(X_train, y_train_np)
        y_pred = est.predict(X_test)
        rep = classification_report(
            y_test_np, y_pred, labels=labels, output_dict=True, zero_division=0, digits=digits
        )
        report_df = pd.DataFrame(rep).transpose()
        report_df.to_csv(fpath, index=True)
        return fname, rep["accuracy"], rep["macro avg"]["f1-score"], rep["weighted avg"]["f1-score"]

    if n_jobs is None or n_jobs == 1:
        rows = [_worker(p) for p in params_list]
    else:
        rows = Parallel(n_jobs=n_jobs)(delayed(_worker)(p) for p in params_list)

    summary_df = pd.DataFrame(rows, columns=["Model", "Accuracy", "Macro Avg F1-Score", "Weighted Avg F1-Score"])
    summary_df.to_csv(output_dir / f"{prefix}_summary_classification_report.csv", index=False)
    return summary_df

def get_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Union[torch.device, str],
    *,
    return_logits: bool = False,
):
    """
    Collect predictions and labels from a dataloader.
    Supports:
      - (inputs, labels)
      - (input_ids, attention_mask, labels)
      - (dict, labels)
    """
    model.eval()
    preds, labels, logits_all = [], [], []

    with torch.inference_mode():
        for batch in dataloader:
            feats, yb = _parse_batch(batch)
            if yb is None:
                raise ValueError("Dataloader must yield labels.")
            out = _forward(model, feats, device)
            pred = torch.argmax(out, dim=1)
            preds.extend(pred.cpu().numpy().tolist())
            labels.extend(yb.cpu().numpy().tolist())
            if return_logits:
                logits_all.append(out.cpu())

    if return_logits:
        return preds, labels, torch.cat(logits_all) if logits_all else None
    return preds, labels

def generate_mlp_reports(
    models: Iterable[Tuple[str, torch.nn.Module]],
    dataloader: torch.utils.data.DataLoader,
    device: Union[torch.device, str],
    output_dir: Path,
    *,
    prefix: str = "MLP",
    weight_dir: Optional[Path] = None,
    labels: Optional[Sequence[int]] = None,
    digits: int = 4,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Evaluate a list of (name, model) on dataloader, write per-model reports, and save a summary CSV.
    If weight_dir is given, loads <name>_best_weights.pth or <name>_weights.pth.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if weight_dir is not None:
        weight_dir = Path(weight_dir)

    rows = []
    for name, model in models:
        if weight_dir is not None:
            best = weight_dir / f"{name}_best_weights.pth"
            std = weight_dir / f"{name}_weights.pth"
            path = best if best.exists() else std
            state = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(state, strict=strict)

        report_path = output_dir / f"{prefix}_{name}.csv"
        if report_path.exists():
            df = pd.read_csv(report_path, index_col=0)
            acc, macro_f1, weighted_f1 = _extract_metrics_from_report_df(df)
        else:
            y_pred, y_true = get_predictions(model, dataloader, device)
            rep = classification_report(
                y_true, y_pred, labels=labels, output_dict=True, zero_division=0, digits=digits
            )
            df = pd.DataFrame(rep).transpose()
            df.to_csv(report_path, index=True)
            acc, macro_f1, weighted_f1 = rep["accuracy"], rep["macro avg"]["f1-score"], rep["weighted avg"]["f1-score"]

        rows.append([name, acc, macro_f1, weighted_f1])

    summary_df = pd.DataFrame(rows, columns=["Model", "Accuracy", "Macro Avg F1-Score", "Weighted Avg F1-Score"])
    summary_df.to_csv(output_dir / f"{prefix}_summary_classification_report.csv", index=False)
    return summary_df

# =======================
# Best-model comparison
# =======================

REPORT_DIRS: Dict[str, str] = {
    "W2V_SVM": "classification_reports_w2v_svm",
    "BERT_SVM": "classification_reports_bert_svm",
    "W2V_MLP": "classification_reports_w2v_mlp",
    "BERT_MLP": "classification_reports_bert_mlp",
}

def _load_summary(dirpath: Path, prefix: str) -> pd.DataFrame:
    cands = [dirpath / f"{prefix}_summary_classification_report.csv", dirpath / "summary_classification_report.csv"]
    for p in cands:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(f"No summary CSV in {dirpath} for prefix '{prefix}'")

def _load_best_full_report(dirpath: Path, prefix: str) -> Tuple[str, pd.DataFrame]:
    summary = _load_summary(dirpath, prefix)
    best = summary.sort_values("Weighted Avg F1-Score", ascending=False).iloc[0]
    model_field = str(best["Model"])
    if model_field.endswith(".csv"):  # SVM summary stores filename
        path = dirpath / model_field
        display_name = Path(model_field).stem
    else:  # MLP summary stores model name
        path = dirpath / f"{prefix}_{model_field}.csv"
        display_name = model_field
    if not path.exists():
        raise FileNotFoundError(f"Report CSV not found: {path}")
    return display_name, pd.read_csv(path, index_col=0)

def build_best_models_comparison(
    files_dir: Path,
    families: Sequence[str] = ("W2V_SVM", "W2V_MLP", "BERT_SVM", "BERT_MLP"),
    class_labels: Sequence[int] = (0, 1, 2, 3, 4),
    headers: str = "full",  # "full" or "simplified"
    metrics: Sequence[str] = ("Precision", "Recall", "F1-Score"),
    save_to: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build a per-class comparison table for the best model from each family, using saved CSVs.
    headers="simplified" uses family names as the top-level headers in the given order.
    """
    files_dir = Path(files_dir)
    base_reports = files_dir / "reports"

    family_to_best: Dict[str, Tuple[str, pd.DataFrame]] = {}
    for fam in families:
        subdir = REPORT_DIRS.get(fam)
        if subdir is None:
            continue
        dirpath = base_reports / subdir
        if not dirpath.exists():
            continue
        try:
            model_name, rep = _load_best_full_report(dirpath, fam)
            family_to_best[fam] = (model_name, rep)
        except FileNotFoundError:
            continue

    if not family_to_best:
        raise RuntimeError("No best reports found. Generate summaries and per-model CSVs first.")

    def _label(fam: str, display: str) -> str:
        return fam if headers == "simplified" else display

    col_keys: List[str] = []
    for fam in families:
        if fam in family_to_best:
            display_name, _ = family_to_best[fam]
            col_keys.append(_label(fam, display_name))

    columns = pd.MultiIndex.from_product([col_keys, list(metrics)])
    idx = list(class_labels)
    comparison_df = pd.DataFrame(index=idx, columns=columns)

    # Support from the first report
    first_report = next(iter(family_to_best.values()))[1]
    support = first_report.loc[first_report.index.astype(str).isin(map(str, class_labels)), "support"]
    support.index = support.index.astype(int)
    comparison_df.insert(0, "Support", support.reindex(class_labels).astype(int).astype(str))

    # Fill metrics
    for fam in families:
        if fam not in family_to_best:
            continue
        display_name, rep = family_to_best[fam]
        hdr = _label(fam, display_name)

        rows = rep.loc[rep.index.astype(str).isin(map(str, class_labels))].copy()
        rows.index = rows.index.astype(int)
        if "Precision" in metrics:
            comparison_df[(hdr, "Precision")] = rows["precision"].astype(float).reindex(class_labels)
        if "Recall" in metrics:
            comparison_df[(hdr, "Recall")] = rows["recall"].astype(float).reindex(class_labels)
        if "F1-Score" in metrics:
            comparison_df[(hdr, "F1-Score")] = rows["f1-score"].astype(float).reindex(class_labels)

    # Averages
    weights = support.reindex(class_labels).astype(float).values
    total_support = int(support.sum())
    normal_avg = {"Support": str(total_support)}
    weighted_avg = {"Support": str(total_support)}

    for fam in families:
        if fam not in family_to_best:
            continue
        display_name, _ = family_to_best[fam]
        hdr = _label(fam, display_name)
        for m in metrics:
            vals = comparison_df.loc[list(class_labels), (hdr, m)].astype(float).values
            normal_avg[(hdr, m)] = float(np.mean(vals))
            weighted_avg[(hdr, m)] = float(np.average(vals, weights=weights))

    comparison_df.loc["Average"] = normal_avg
    comparison_df.loc["Weighted Average"] = weighted_avg

    # Round numeric cells (keep 'Support' as strings)
    for col in comparison_df.columns:
        if col != "Support":
            comparison_df[col] = pd.to_numeric(comparison_df[col], errors="coerce").round(2)

    ordered = list(class_labels) + ["Average", "Weighted Average"]
    comparison_df = comparison_df.reindex(ordered)

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(save_to, index=True)

    return comparison_df

# =======================
# Display helpers
# =======================

def display_report(
    df: pd.DataFrame,
    sort_col: str = "Weighted Avg F1-Score",
    ascending: bool = False,
    top: Optional[int] = None,
    decimals: int = 4,
):
    """Pretty, notebook-friendly table for flat summaries."""
    out = df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    if top is not None:
        out = out.head(top)
    fmt = {c: f"{{:.{decimals}f}}" for c in out.select_dtypes(include="number").columns}
    try:
        from IPython.display import display
        display(out.style.format(fmt))
    except Exception:
        print(out.to_string(index=False, formatters=fmt))

def display_comparison(df: pd.DataFrame, center_headers: bool = True, decimals: int = 2):
    """Pretty display for the multiindex comparison table."""
    fmt = {c: f"{{:.{decimals}f}}" for c in df.columns if c != "Support"}
    try:
        from IPython.display import display
        styler = df.style.format(fmt)
        if center_headers and isinstance(df.columns, pd.MultiIndex):
            styler = styler.set_table_styles(
                [
                    {"selector": "th.col_heading.level0", "props": [("text-align", "center")]},
                    {"selector": "th.col_heading.level1", "props": [("text-align", "center")]},
                    {"selector": "th.row_heading", "props": [("text-align", "right")]},
                ],
                overwrite=False,
            )
        display(styler)
    except Exception:
        print(df.to_string())

__all__ = [
    "evaluate",
    "get_predictions",
    "generate_svm_reports",
    "generate_mlp_reports",
    "build_best_models_comparison",
    "display_report",
    "display_comparison",
]