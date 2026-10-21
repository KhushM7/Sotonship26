import os
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]

# the data folder; the webcam demo needs none, so a fresh clone runs without it
DATA = Path(os.environ.get("FALL_DATA") or _REPO_ROOT / "data")


def load_config(
    path: str | os.PathLike[str] = "configs/default.yaml",
) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = _REPO_ROOT / p
    with open(p, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    return cfg


def skeleton_path(dataset: str, path: str) -> Path:
    """Contract 1: data/skeletons/{dataset}/{path}.npz (path may contain sub-folders)."""
    return DATA / "skeletons" / dataset / f"{path}.npz"


def scores_path(model: str, split: str) -> Path:
    """Contract 4: reports/scores/{model}/{split}.parquet (repo-relative)."""
    return _REPO_ROOT / "reports" / "scores" / model / f"{split}.parquet"


def windows_path(split: str) -> Path:
    """Contract 3: data/windows/{split}.npz."""
    return DATA / "windows" / f"{split}.npz"
