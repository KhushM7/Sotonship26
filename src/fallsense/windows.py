"""Contract 3 v2: window arrays, data/windows/{split}.npz, and the windowing rules."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from fallsense.config import load_config, windows_path

N_CLASSES = 4
WINDOW = 30
N_KEYPOINTS = 13
N_CTX = 4  # occlusion.last_seen_context columns
FALLING = 1

# key -> (dtype, shape after the leading N axis)
SCHEMA = {
    "kp": (np.float32, (WINDOW, N_KEYPOINTS, 3)),
    "wh": (np.int16, (2,)),
    "y": (np.int64, ()),
    "clip_id": (np.int32, ()),
    "end_frame": (np.int32, ()),
    "present": (np.bool_, (WINDOW,)),
    "ctx": (np.float32, (WINDOW, N_CTX)),
}

_CFG = load_config()
# positions of the hips within the 13 config keypoints
_L_HIP, _R_HIP = 7, 8


def validate_windows(arrays: dict[str, npt.NDArray[Any]]) -> None:
    """Raise ValueError unless `arrays` matches Contract 3."""
    missing = set(SCHEMA) - set(arrays)
    if missing:
        raise ValueError(f"missing keys: {sorted(missing)}")
    n = len(arrays["y"])
    for key, (dtype, tail) in SCHEMA.items():
        a = arrays[key]
        if a.dtype != dtype:
            raise ValueError(f"{key}: dtype {a.dtype}, expected {np.dtype(dtype)}")
        if a.shape != (n, *tail):
            raise ValueError(f"{key}: shape {a.shape}, expected {(n, *tail)}")
    if n and not ((arrays["y"] >= 0) & (arrays["y"] < N_CLASSES)).all():
        raise ValueError(f"y must be in 0..{N_CLASSES - 1}")
    if not np.isfinite(arrays["ctx"]).all():
        raise ValueError("ctx must be finite")


def save_windows(
    split: str,
    kp: npt.ArrayLike,
    wh: npt.ArrayLike,
    y: npt.ArrayLike,
    clip_id: npt.ArrayLike,
    end_frame: npt.ArrayLike,
    present: npt.ArrayLike,
    ctx: npt.ArrayLike,
    path: Path | None = None,
) -> None:
    """Cast to the contract dtypes, validate and write data/windows/{split}.npz.

    `path` overrides the contract location (dry runs only).
    """
    arrays: dict[str, npt.NDArray[Any]] = {
        key: np.asarray(val, dtype=SCHEMA[key][0])
        for key, val in zip(SCHEMA, (kp, wh, y, clip_id, end_frame, present, ctx))
    }
    validate_windows(arrays)
    path = windows_path(split) if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **cast("dict[str, Any]", arrays))


def load_windows(split: str, path: Path | None = None) -> dict[str, npt.NDArray[Any]]:
    """Load and validate data/windows/{split}.npz (`path` overrides, dry runs only)."""
    with np.load(windows_path(split) if path is None else path) as d:
        arrays = {key: d[key] for key in SCHEMA}
    validate_windows(arrays)
    return arrays


# ---------------------------------------------------------------- per clip


def select_keypoints(
    kp33: np.ndarray, detected: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Contract 1 (T, 33, 4) normalised -> (T, 13, 3) pixel x, y, visibility.

    NaN wherever no pose was detected.
    """
    kp = kp33[:, _CFG["keypoints"]][..., [0, 1, 3]].astype(np.float32)
    kp[..., 0] *= width
    kp[..., 1] *= height
    kp[~np.asarray(detected, dtype=bool)] = np.nan
    return kp


def _fallback_last_seen_context(
    kp: np.ndarray, present: np.ndarray, wh: np.ndarray, fps: float, cap_s: float
) -> np.ndarray:
    """Spec copy of Dev A's occlusion.last_seen_context, used only until M19 lands.

    Columns: seconds since present (capped), last hip height (y / H), last hip
    vertical speed (frame heights/s over the last 5 present frames, + = down),
    last hip distance to the nearest border (0-0.5).
    """
    w, h = float(wh[0]), float(wh[1])
    hip = (kp[:, _L_HIP, :2] + kp[:, _R_HIP, :2]) / 2
    out = np.zeros((len(present), N_CTX), dtype=np.float32)
    since, last = cap_s, np.zeros(3)
    seen_t: list[int] = []
    seen_y: list[float] = []
    for t in range(len(present)):
        if present[t] and np.isfinite(hip[t]).all():
            x, y = hip[t] / (w, h)
            seen_t.append(t)
            seen_y.append(float(y))
            t0, y0 = seen_t[-5:][0], seen_y[-5:][0]
            speed = (y - y0) * fps / (t - t0) if t > t0 else 0.0
            edge = min(x, 1 - x, y, 1 - y)
            since, last = 0.0, np.array([y, speed, np.clip(edge, 0, 0.5)])
        else:
            since = min(since + 1 / fps, cap_s)
        out[t] = (since, *last)
    return out


def last_seen_context(
    kp: np.ndarray, present: np.ndarray, wh: np.ndarray, fps: float, cap_s: float
) -> np.ndarray:
    """Dev A's occlusion.last_seen_context when merged, else the spec fallback."""
    try:
        from fallsense.occlusion import last_seen_context as dev_a
    except ImportError:
        return _fallback_last_seen_context(kp, present, wh, fps, cap_s)
    return np.asarray(dev_a(kp, present, wh, fps, cap_s), dtype=np.float32)


def window_label(y4: np.ndarray, recent: np.ndarray) -> int | None:
    """Label a window by what happens at its end; None drops it.

    `recent` is "present or recently seen": present | (ctx[:, 0] < ctx_cap_s),
    so a person hidden after a fall keeps the window (Contract 3 v2).
    """
    if (y4 < 0).mean() > 0.30 or (~recent).mean() > 0.40:
        return None
    if (y4[-20:] == FALLING).sum() >= 5:
        return FALLING
    tail = y4[-8:]
    tail = tail[tail >= 0]
    return int(np.bincount(tail, minlength=N_CLASSES).argmax()) if len(tail) else None


def window_ends(
    n_frames: int, y4: np.ndarray, train: bool, cfg: dict = _CFG
) -> np.ndarray:
    """Last frame of every window: stride_train (plus stride_train_near_fall within
    1 s of a falling frame) for train, stride_eval otherwise (val, test, OOPS)."""
    first = cfg["window"] - 1
    if n_frames <= first:
        return np.zeros(0, dtype=np.int64)
    if not train:
        return np.arange(first, n_frames, cfg["stride_eval"])
    ends = set(range(first, n_frames, cfg["stride_train"]))
    fall = np.flatnonzero(y4 == FALLING)
    if len(fall):
        near = np.arange(first, n_frames, cfg["stride_train_near_fall"])
        # distance from each candidate end to the nearest falling frame
        dist = np.abs(near[:, None] - fall[None, :]).min(axis=1)
        ends.update(near[dist <= cfg["fps"]].tolist())
    return np.array(sorted(ends), dtype=np.int64)


def clip_windows(
    kp: np.ndarray,
    present: np.ndarray,
    ctx: np.ndarray,
    y4: np.ndarray,
    ends: np.ndarray,
    cap_s: float,
) -> dict[str, np.ndarray]:
    """Cut one clip into labelled windows; returns kp, present, ctx, y, end_frame."""
    n = WINDOW
    recent = present | (ctx[:, 0] < cap_s)
    keep_e, keep_y = [], []
    for e in ends:
        s = e - n + 1
        if s < 0 or e >= len(y4):
            raise ValueError(f"window [{s}, {e}] crosses the clip boundary")
        lab = window_label(y4[s : e + 1], recent[s : e + 1])
        if lab is not None:
            keep_e.append(e)
            keep_y.append(lab)
    idx = np.array(keep_e, dtype=np.int64)[:, None] + np.arange(-n + 1, 1)
    return {
        "kp": kp[idx] if len(keep_e) else np.zeros((0, n, N_KEYPOINTS, 3), np.float32),
        "present": present[idx] if len(keep_e) else np.zeros((0, n), bool),
        "ctx": ctx[idx] if len(keep_e) else np.zeros((0, n, N_CTX), np.float32),
        "y": np.array(keep_y, dtype=np.int64),
        "end_frame": np.array(keep_e, dtype=np.int64),
    }


# ---------------------------------------------------------------- leak checks


def check_splits(
    index: pd.DataFrame, exempt_datasets: tuple[str, ...] = ("mcfd",)
) -> None:
    """Raise ValueError if a clip is in two splits or a person is in train and test.

    `subject == -1` (unknown) and single-actor datasets are exempt from the
    person check.
    """
    per_clip = index.groupby(["dataset", "path"])["split"].nunique()
    if (per_clip > 1).any():
        raise ValueError(
            f"clips in two splits: {per_clip[per_clip > 1].index.tolist()[:5]}"
        )
    known = index[(index["subject"] != -1) & ~index["dataset"].isin(exempt_datasets)]
    train = set(
        map(tuple, known.loc[known["split"] == "train", ["dataset", "subject"]].values)
    )
    test = set(
        map(tuple, known.loc[known["split"] == "test", ["dataset", "subject"]].values)
    )
    leaked = train & test
    if leaked:
        raise ValueError(f"(dataset, subject) in train and test: {sorted(leaked)[:5]}")
