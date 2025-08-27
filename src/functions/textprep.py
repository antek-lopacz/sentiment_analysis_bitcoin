from __future__ import annotations

from typing import Iterable, List, Sequence, Union, Tuple, Optional
import re
import string
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# NLTK init and caches
# -----------------------------
_NLTK_READY = False
_STOP_WORDS: Optional[set] = None
_LEMMATIZER: Optional[WordNetLemmatizer] = None

def _ensure_nltk_ready() -> None:
    """Download required NLTK data (once) and cache stopwords + lemmatizer."""
    global _NLTK_READY, _STOP_WORDS, _LEMMATIZER
    if _NLTK_READY:
        return
    for pkg in ("punkt", "stopwords", "wordnet", "omw-1.4"):
        try:
            if pkg == "punkt":
                nltk.data.find("tokenizers/punkt")
            else:
                nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    _STOP_WORDS = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()
    _NLTK_READY = True

# -----------------------------
# Regexes compiled once
# -----------------------------
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMOJI_PATTERN = re.compile(
    "["                       # start class
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map symbols
    "\U0001F1E0-\U0001F1FF"   # flags
    "\u2600-\u26FF"           # misc symbols
    "\u2700-\u27BF"           # dingbats
    "\U0001F900-\U0001F9FF"   # supplemental symbols & pictographs
    "\U0001FA70-\U0001FAFF"   # symbols & pictographs extended-A
    "]+",
    flags=re.UNICODE,
)

# -----------------------------
# Tokenization & cleaning
# -----------------------------
def tokenize_function(text: str) -> List[str]:
    """Tokenize text into words using NLTK (Punkt)."""
    _ensure_nltk_ready()
    return nltk.word_tokenize(text)

def clean_body(
    body: str,
    *,
    remove_urls: bool = True,
    remove_emojis: bool = True,
) -> str:
    """Lowercase, replace URLs with <URL>, remove emojis, collapse whitespace."""
    s = body.lower()
    if remove_urls:
        s = URL_PATTERN.sub("<URL>", s)
    if remove_emojis:
        s = EMOJI_PATTERN.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_tokens(
    tokens: Iterable[str],
    *,
    custom_words_to_keep: Iterable[str] = (),
    lemmatize: bool = True,
) -> List[str]:
    """
    Lowercase, remove stopwords/punctuation, map URLs-><URL>, remove emojis,
    (optionally) lemmatize, keep placeholders like <URL>, <NUM>, <YEAR>.
    """
    _ensure_nltk_ready()
    sw = set(_STOP_WORDS)  # type: ignore
    if custom_words_to_keep:
        sw -= set(custom_words_to_keep)

    out: List[str] = []
    for tok in tokens:
        t = tok.lower()

        # URL placeholder
        if URL_PATTERN.match(t):
            out.append("<URL>")
            continue

        # strip emojis within token
        t = EMOJI_PATTERN.sub("", t)
        if not t:
            continue

        # pure punctuation -> drop
        if t in string.punctuation:
            continue

        # stopwords
        if t in sw:
            continue

        # lemmatize words (but not placeholders like <URL>)
        if lemmatize and not (t.startswith("<") and t.endswith(">")):
            t = _LEMMATIZER.lemmatize(t)  # type: ignore

        # keep alnum or '<PLACEHOLDER>'
        if t.isalnum() or (t.startswith("<") and t.endswith(">")):
            out.append(t)

    return out

# numeric placeholders
NUM_PRC_PATTERN = re.compile(r"^[0-9]+(?:[.,][0-9]+)?\s*%$")
YEAR_PATTERN    = re.compile(r"^20\d{2}$")
NUM_PATTERN     = re.compile(r"^[0-9][0-9.,]*$")

def replace_numerical_tokens(tokens: Iterable[str]) -> List[str]:
    """Replace numeric-like tokens with placeholders: <NUM_PRC>, <YEAR>, <NUM>."""
    new_tokens: List[str] = []
    for t in tokens:
        if NUM_PRC_PATTERN.match(t):
            new_tokens.append("<NUM_PRC>")
        elif YEAR_PATTERN.match(t):
            new_tokens.append("<YEAR>")
        elif NUM_PATTERN.match(t):
            new_tokens.append("<NUM>")
        else:
            new_tokens.append(t)
    return new_tokens

def preprocess_tokens(
    text: str,
    *,
    custom_words_to_keep: Iterable[str] = (),
    replace_numbers: bool = True,
) -> List[str]:
    """Convenience pipeline: tokenize -> clean -> (optional) replace numerics."""
    toks = tokenize_function(text)
    toks = clean_tokens(toks, custom_words_to_keep=custom_words_to_keep)
    if replace_numbers:
        toks = replace_numerical_tokens(toks)
    return toks

# -----------------------------
# Word2Vec sentence embedding
# -----------------------------
def post_to_embedding(
    tokens: Iterable[str],
    word2vec,
    *,
    embedding_dim: Optional[int] = None,
    agg: str = "mean",
    normalize: bool = False,
) -> np.ndarray:
    """
    Average word vectors for tokens present in the model.
    - embedding_dim: defaults to word2vec.vector_size
    - agg: 'mean' (future: 'median', 'max' if needed)
    - normalize: L2 normalize the final vector
    """
    if embedding_dim is None:
        embedding_dim = int(getattr(word2vec, "vector_size", 300))
    toks = list(tokens)
    if not toks:
        vec = np.zeros(embedding_dim, dtype=np.float32)
        return vec

    # ensure numeric placeholders are present if upstream pipeline omitted it
    toks = replace_numerical_tokens(toks)

    vecs = [word2vec[t] for t in toks if t in getattr(word2vec, "key_to_index", {})]
    if not vecs:
        vec = np.zeros(embedding_dim, dtype=np.float32)
    else:
        arr = np.vstack(vecs).astype(np.float32)
        if agg == "mean":
            vec = arr.mean(axis=0)
        else:
            vec = arr.mean(axis=0)  # safe default

    if normalize:
        n = np.linalg.norm(vec) + 1e-12
        vec = vec / n
    return vec

# -----------------------------
# BERT helpers
# -----------------------------
def bert_tokenize_function(
    text: Union[str, Sequence[str]],
    bert_tokenizer,
    *,
    max_length: int = 256,
    pad_to_max_length: bool = True,
    return_tensors: Optional[str] = "pt",
):
    """Tokenize with HF tokenizer; padding='max_length' when pad_to_max_length else to longest."""
    padding = "max_length" if pad_to_max_length else True
    return bert_tokenizer(
        text,
        padding=padding,
        truncation=True if max_length else False,
        max_length=max_length if max_length else None,
        return_tensors=return_tensors,
    )

def get_bert_embedding(
    inputs,                       # dict with 'input_ids' and 'attention_mask' (tensors)
    *,
    model,                        # pass a preloaded HF model (e.g., BertModel)
    device: str = "cpu",
    pool: str = "cls",            # 'cls' or 'mean'
) -> np.ndarray:
    """
    Forward pass with a preloaded model; returns (batch, hidden) numpy array.
    No model instantiation here.
    """
    import torch
    with torch.no_grad():
        ids = inputs["input_ids"].to(device)
        mask = inputs.get("attention_mask", None)
        mask = mask.to(device) if mask is not None else None
        outputs = model(input_ids=ids, attention_mask=mask)
        last_hidden = outputs.last_hidden_state  # (B, T, H)

        if pool == "cls":
            pooled = last_hidden[:, 0, :]        # (B, H)
        elif pool == "mean":
            # mean over valid tokens using mask if provided
            if mask is None:
                pooled = last_hidden.mean(dim=1)
            else:
                mask_f = mask.float().unsqueeze(-1)  # (B, T, 1)
                summed = (last_hidden * mask_f).sum(dim=1)
                counts = mask_f.sum(dim=1).clamp(min=1.0)
                pooled = summed / counts
        else:
            raise ValueError("pool must be 'cls' or 'mean'.")

    return pooled.cpu().numpy()

__all__ = [
    "tokenize_function",
    "clean_body",
    "clean_tokens",
    "replace_numerical_tokens",
    "preprocess_tokens",
    "post_to_embedding",
    "bert_tokenize_function",
    "get_bert_embedding",
]