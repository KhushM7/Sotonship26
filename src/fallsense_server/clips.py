"""Replay source: a recorded clip's skeletons plus the backend's stored scores and alerts.

Skeletons are Contract 1, scores are Contract 4 (reports/scores/<run>/<split>.parquet)
and alerts come from `fallsense.calibrate.alerts_from_scores` with the run's
thresholds.yaml, so a replay room shows exactly what the backend decided.

A 09_live.py recording (`{recording: <folder> | latest}` in rooms.yaml) plays the
same way, with the scores and alerts the live script decided while it recorded.
"""

from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fallsense.calibrate import CLASS_COLUMNS, alerts_from_scores
from fallsense.config import DATA, load_config, skeleton_path
from fallsense_server.settings import backend_root
from fallsense_server.videos import (
    RECORDING,
    latest_recording,
    recordings_dir,
    video_url,
)

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
        d = dict(d)
        if "recording" in d:  # path "latest" is resolved when the clip starts
            d = {"dataset": RECORDING, "path": str(d.pop("recording")), **d}
        return cls(**d)

    @property
    def is_recording(self) -> bool:
        return self.dataset == RECORDING

    def resolved(self) -> "ClipRef | None":
        """This ref with `latest` replaced by the newest recording (None if none yet)."""
        if not (self.is_recording and self.path == "latest"):
            return self
        name = latest_recording()
        return None if name is None else replace(self, path=name)


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
    video: str | None = None  # address of the source video; None = skeleton only
    video_t0: float = 0.0  # seconds into the video of the clip's first frame

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
    if ref.is_recording:
        return load_recording(ref)
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

    kp, present, end_frame, probs, held_from = _hold(ref, kp, present, end_frame, probs)
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
        video=video_url(ref.dataset, ref.path),
        video_t0=f0 / FPS,
    )


def _hold(
    ref: ClipRef,
    kp: np.ndarray,
    present: np.ndarray,
    end_frame: np.ndarray,
    probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float | None]:
    """Keep the last pose and scores for ref.hold_s after the clip ends."""
    if ref.hold_s <= 0 or not len(kp) or not len(end_frame):
        return kp, present, end_frame, probs, None
    held_from = len(kp) / FPS
    n_hold = round(ref.hold_s * FPS)
    kp = np.concatenate([kp, np.repeat(kp[-1:], n_hold, axis=0)])
    present = np.concatenate([present, np.repeat(present[-1:], n_hold)])
    extra = np.arange(end_frame[-1] + EVAL_STRIDE, len(kp), EVAL_STRIDE)
    end_frame = np.concatenate([end_frame, extra])
    probs = np.concatenate([probs, np.repeat(probs[-1:], len(extra), axis=0)])
    return kp, present, end_frame, probs, held_from


def load_recording(ref: ClipRef) -> Clip:
    """A 09_live.py --record session: its skeletons, and the scores and alerts the
    live script decided (scores.csv, alerts.csv), so no model runs here."""
    folder = recordings_dir() / ref.path
    with np.load(folder / "skeleton.npz") as d:
        kp33 = d["kp"].astype(np.float32)
        detected = d["detected"].astype(bool)
        aspect = float(d["width"]) / float(d["height"])
    kp = kp33[:, KEYPOINTS][:, :, [0, 1, 3]] if len(kp33) else np.zeros((0, 13, 3))
    present = detected & np.isfinite(kp[:, :, 0]).any(axis=1)
    scores = pd.read_csv(folder / "scores.csv")
    alerts = pd.read_csv(folder / "alerts.csv")
    f0 = round(ref.start_s * FPS)
    f1 = len(kp) if ref.end_s is None else min(len(kp), round(ref.end_s * FPS))
    kp, present = kp[f0:f1], present[f0:f1]
    end_frame = np.round(scores["t_s"].to_numpy(np.float64) * FPS).astype(np.int64)
    keep = (end_frame >= f0) & (end_frame < f1)
    # scores.csv columns are t_s then the config classes, i.e. CLASS_COLUMNS order
    probs = scores.iloc[:, 1:5].to_numpy(np.float64)[keep]
    end_frame = end_frame[keep] - f0
    kp, present, end_frame, probs, held_from = _hold(ref, kp, present, end_frame, probs)
    t0, t1 = f0 / FPS, f1 / FPS
    return Clip(
        ref=ref,
        kp=kp,
        present=present,
        win_t=end_frame / FPS,
        probs=probs,
        alerts=[
            (float(r.t_s) - t0, str(r.level), float(r.score))
            for r in alerts.itertuples()
            if t0 <= r.t_s < t1
        ],
        aspect=aspect,
        held_from_s=held_from,
        video=video_url(ref.dataset, ref.path),
        video_t0=t0,
    )


def clip_exists(ref: ClipRef) -> bool:
    if ref.is_recording:
        r = ref.resolved()
        return r is not None and (recordings_dir() / r.path / "session.yaml").exists()
    return Path(skeleton_path(ref.dataset, ref.path)).exists()
