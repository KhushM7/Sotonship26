import os
from pathlib import Path

import yaml

DATA = Path(os.environ["FALL_DATA"])

_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path="configs/default.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = _REPO_ROOT / p
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def skeleton_path(dataset, path) -> Path:
    """Contract 1: data/skeletons/{dataset}/{path}.npz (path may contain sub-folders)."""
    return DATA / "skeletons" / dataset / f"{path}.npz"


def windows_path(split) -> Path:
    """Contract 3: data/windows/{split}.npz."""
    return DATA / "windows" / f"{split}.npz"
