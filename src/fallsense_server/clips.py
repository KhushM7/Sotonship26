"""Replay source: a recorded clip's skeletons plus the backend's stored scores and alerts.

Skeletons are Contract 1, scores are Contract 4 (reports/scores/<run>/<split>.parquet)
and alerts come from `fallsense.calibrate.alerts_from_scores` with the run's
thresholds.yaml, so a replay room shows exactly what the backend decided.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fallsense.calibrate import CLASS_COLUMNS, alerts_from_scores
from fallsense.config import DATA, load_config, skeleton_path
from fallsense_server.settings import backend_root

CFG = load_config()
FPS = float(CFG["fps"])
KEYPOINTS = list(CFG["keypoints"])
EVAL_STRIDE = int(CFG["stride_eval"])
# the 13 config keypoints: nose, shoulders, elbows, wrists, hips, knees, ankles
BONES = [
    (1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8),
    (7, 8), (7, 9), (9, 11), (8, 10), (10, 12),
]  # fmt: skip


@dataclass(frozen=True)
class ClipRef:
    """One entry of a room's playlist in configs/rooms.yaml."""

    dataset: str
    path: str
    start_s: float = 0.0
    end_s: float | None = None
    hold_s: float = 0.0  # keep the last pose and scores this long after the clip ends
    split: str | None = None

    @classmethod
    def from_cfg(cls, d: dict[str, Any]) -> "ClipRef":
        return cls(**d)


@dataclass
class Clip:
    ref: ClipRef
    kp: np.ndarray  # (T, 13, 3) normalised x, y, visibility; NaN where lost
    present: np.ndarray  # (T,) bool
    win_t: np.ndarray  # (N,) seconds from clip start, one per evaluation window
    probs: np.ndarray  # (N, 4) in CLASS_COLUMNS order
    alerts: list[tuple[float, str, float]] = field(default_factory=list)
    aspect: float = 16 / 9  # source width / height, to draw without distortion
    held_from_s: float | None = None  # start of the held tail, if any

    @property
    def duration_s(self) -> float:
        return len(self.kp) / FPS


def load_thresholds(run: str) -> dict[str, Any]:
    """models/<run>/thresholds.yaml, else reports/scores/<run>/thresholds.yaml."""
    root = backend_root()
    for path in (
        root / "models" / run / "thresholds.yaml",
        root / "reports" / "scores" / run / "thresholds.yaml",
    ):
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    raise FileNotFoundError(f"no thresholds.yaml for run {run} under {root}")


def on_floor_seconds(th: dict[str, Any]) -> float:
    return float(th.get("on_floor_seconds_demo", CFG["on_floor_seconds"]))


@lru_cache(maxsize=1)
def _index() -> pd.DataFrame:
    return pd.read_csv(DATA / "skeletons" / "index.csv")


@lru_cache(maxsize=8)
def _scores(run: str, split: str) -> pd.DataFrame:
    path = backend_root() / "reports" / "scores" / run / f"{split}.parquet"
    return pd.read_parquet(path)


def clip_id(ref: ClipRef) -> tuple[int, str]:
    """Contract 2 row number (the clip_id in Contract 4) and the clip's split."""
    idx = _index()
    hit = idx.index[(idx["dataset"] == ref.dataset) & (idx["path"] == ref.path)]
    if len(hit) == 0:
        raise KeyError(f"{ref.dataset}/{ref.path} is not in index.csv")
    i = int(hit[0])
    return i, ref.split or str(idx.at[i, "split"])


def load_clip(ref: ClipRef, run: str) -> Clip:
    cid, split = clip_id(ref)
    with np.load(skeleton_path(ref.dataset, ref.path)) as d:
        kp33 = d["kp"].astype(np.float32)
        detected = d["detected"].astype(bool)
    kp = kp33[:, KEYPOINTS][:, :, [0, 1, 3]]  # x, y, visibility
    present = detected & np.isfinite(kp[:, :, 0]).any(axis=1)

    rows = _scores(run, split)
    rows = rows[rows["clip_id"] == cid].sort_values("end_frame")
    if rows.empty:
        raise KeyError(f"no {run} scores for {ref.dataset}/{ref.path} in {split}")

    f0 = round(ref.start_s * FPS)
    f1 = len(kp) if ref.end_s is None else min(len(kp), round(ref.end_s * FPS))
    kp, present = kp[f0:f1], present[f0:f1]
    rows = rows[(rows["end_frame"] >= f0) & (rows["end_frame"] < f1)]
    end_frame = rows["end_frame"].to_numpy(np.int64) - f0
    probs = rows[CLASS_COLUMNS].to_numpy(np.float64)

    held_from = None
    if ref.hold_s > 0 and len(kp) and len(end_frame):
        held_from = len(kp) / FPS
        n_hold = round(ref.hold_s * FPS)
        kp = np.concatenate([kp, np.repeat(kp[-1:], n_hold, axis=0)])
        present = np.concatenate([present, np.repeat(present[-1:], n_hold)])
        extra = np.arange(end_frame[-1] + EVAL_STRIDE, len(kp), EVAL_STRIDE)
        end_frame = np.concatenate([end_frame, extra])
        probs = np.concatenate([probs, np.repeat(probs[-1:], len(extra), axis=0)])

    win_t = end_frame / FPS
    th = load_thresholds(run)
    df = pd.DataFrame({"clip_id": 0, "end_frame": end_frame, "t_s": win_t})
    df[CLASS_COLUMNS] = probs
    alerts = alerts_from_scores(df, th, on_floor_seconds=on_floor_seconds(th))
    return Clip(
        ref=ref,
        kp=kp,
        present=present,
        win_t=win_t,
        probs=probs,
        alerts=[
            (float(r.t_s), str(r.level), float(r.score)) for r in alerts.itertuples()
        ],
        held_from_s=held_from,
        aspect=float(_index().at[cid, "width"]) / float(_index().at[cid, "height"]),
    )


def clip_exists(ref: ClipRef) -> bool:
    return Path(skeleton_path(ref.dataset, ref.path)).exists()
