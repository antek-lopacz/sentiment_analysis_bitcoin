from pathlib import Path

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

DATA_DIR = repo_root() / "data"
FILES_DIR = repo_root() / "files"
TSB_DIR = FILES_DIR / "tensorboard"
