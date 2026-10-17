"""Training dataset: augment raw window keypoints, then make_features.

Augmentation happens on the (30, 13, 3) pixel keypoints so every augmented
example goes through the same make_features as evaluation and the live demo.
Simulated occlusion (occlusion handling part 3) needs a ClipBank of whole train
clips and Dev A's occlusion.simulate_occlusion; it is off unless asked for.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fallsense import labels, windows
from fallsense.config import load_config, skeleton_path
from fallsense.features import N_FEATURES, make_features, torso_length

CFG = load_config()
OCC = CFG["occlusion"]
FPS = CFG["fps"]
WINDOW = CFG["window"]
L_HIP, R_HIP = 7, 8
# left/right keypoint pairs within the 13 config keypoints
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
FALL_CLASSES = (1, 2)  # falling, fallen

SimulateFn = Callable[..., tuple[np.ndarray, np.ndarray]]
Window = tuple[np.ndarray, np.ndarray, np.ndarray]  # kp, present, ctx


# ---------------------------------------------------------------- augmentation


def _hip_centre(kp: np.ndarray, wh: np.ndarray) -> np.ndarray:
    hip = (kp[:, L_HIP, :2] + kp[:, R_HIP, :2]) / 2
    ok = np.isfinite(hip).all(axis=1)
    return hip[ok].mean(axis=0) if ok.any() else np.asarray(wh, np.float64) / 2


def _speed_change(
    kp: np.ndarray, present: np.ndarray, ctx: np.ndarray, rate: float
) -> Window:
    """Resample in time by `rate`, anchored at the last frame."""
    n = len(kp)
    src = np.clip((n - 1) - (n - 1 - np.arange(n)) * rate, 0, n - 1)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (src - lo)[:, None]
    kp = kp[lo] * (1 - w[:, :, None]) + kp[hi] * w[:, :, None]
    ctx = ctx[lo] * (1 - w) + ctx[hi] * w
    present = present[np.rint(src).astype(int)]
    return kp, present, ctx


def augment_kp(
    kp: np.ndarray,
    present: np.ndarray,
    ctx: np.ndarray,
    wh: np.ndarray,
    rng: np.random.Generator,
) -> Window:
    """The plan's augmentation table on one (30, 13, 3) pixel window."""
    kp = kp.astype(np.float64)
    present, ctx = present.copy(), ctx.astype(np.float64)
    w, h = float(wh[0]), float(wh[1])

    if rng.random() < 0.5:  # mirror left-right
        kp[..., 0] = w - kp[..., 0]
        for a, b in LR_PAIRS:
            kp[:, [a, b]] = kp[:, [b, a]]

    c = _hip_centre(kp, wh)
    xy = kp[..., :2] - c
    if rng.random() < 0.5:  # in-plane rotation, camera roll
        t = np.deg2rad(rng.uniform(-10, 10))
        xy = xy @ np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    if rng.random() < 0.5:  # scale, distance to camera
        xy = xy * rng.uniform(0.85, 1.15)
    xy = xy + c
    if rng.random() < 0.5:  # translation, position in room
        xy = xy + rng.uniform(-0.1, 0.1, size=2) * (w, h)

    vis = np.nan_to_num(kp[..., 2])
    L = torso_length(xy, vis, h)
    xy = xy + rng.normal(0, 0.01 * L, size=xy.shape)  # pose noise
    kp[..., :2] = xy

    kp[rng.random(kp.shape[:2]) < 0.05] = np.nan  # joint dropout
    lost = rng.random(len(kp)) < 0.03  # whole-frame dropout
    kp[lost] = np.nan
    present[lost] = False

    if rng.random() < 0.3:  # speed change
        kp, present, ctx = _speed_change(kp, present, ctx, rng.uniform(0.8, 1.2))
    return kp.astype(np.float32), present, ctx.astype(np.float32)


# ---------------------------------------------------------------- simulated occlusion


class ClipBank:
    """Whole train clips in memory (13-keypoint pixels, present, 4-class labels)."""

    def __init__(self, clips: dict[int, dict[str, np.ndarray]]) -> None:
        self.clips = clips

    @classmethod
    def from_index(cls, index: pd.DataFrame, clip_ids: np.ndarray) -> "ClipBank":
        clips = {}
        for cid in np.unique(clip_ids):
            row = index.iloc[int(cid)]
            d = np.load(skeleton_path(row["dataset"], row["path"]))
            width, height = int(d["width"]), int(d["height"])
            segs = labels.clean_segments(
                labels.segments_for(row["dataset"], row["path"]),
                float(row["duration_s"]),
            )
            clips[int(cid)] = {
                "kp": windows.select_keypoints(d["kp"], d["detected"], width, height),
                "present": d["detected"].astype(bool),
                "y4": labels.to4(labels.frame_labels(segs, d["t_ms"])),
                "wh": np.array([width, height]),
            }
        return cls(clips)


def maybe_occlude(
    bank: ClipBank,
    clip_id: int,
    end_frame: int,
    rng: np.random.Generator,
    simulate: SimulateFn,
    cfg: dict = OCC,
) -> Window | None:
    """Occlude the window ending at `end_frame` with the plan's probabilities.

    Returns the new (kp, present, ctx) for the last 30 frames, or None if this
    window is left alone. The label never changes.
    """
    clip = bank.clips[clip_id]
    s = max(0, end_frame - cfg["lookback_frames"] + 1)
    kp = clip["kp"][s : end_frame + 1]
    present = clip["present"][s : end_frame + 1]
    y4 = clip["y4"][s : end_frame + 1]

    fall = np.flatnonzero(np.isin(y4, FALL_CLASSES))
    p = cfg["aug_p_fall"] if len(fall) else cfg["aug_p_other"]
    if rng.random() >= p:
        return None
    # for falls: between the fall start and the window end; otherwise anywhere
    onset = int(rng.integers(fall[0] if len(fall) else 0, len(kp)))
    mode = "rise" if rng.random() < 0.5 else "box"
    kp, present = simulate(kp.copy(), present.copy(), clip["wh"], rng, onset, mode, cfg)
    ctx = windows.last_seen_context(kp, present, clip["wh"], FPS, cfg["ctx_cap_s"])
    return kp[-WINDOW:], present[-WINDOW:], ctx[-WINDOW:]


def _dev_a_simulate() -> SimulateFn:
    try:
        from fallsense.occlusion import simulate_occlusion
    except ImportError as e:
        raise RuntimeError(
            "simulated occlusion needs Dev A's occlusion.py (M19); train without it"
        ) from e
    return simulate_occlusion


# ---------------------------------------------------------------- dataset


def window_features(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """(N, 30, 51) features for every window, no augmentation (z-score stats, eval)."""
    out = np.empty((len(arrays["y"]), WINDOW, N_FEATURES), np.float32)
    for i in range(len(out)):
        out[i] = make_features(
            arrays["kp"][i], arrays["wh"][i], arrays["present"][i], arrays["ctx"][i]
        )
    return out


class WindowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        npz_path: Path,
        stats: dict[str, np.ndarray],
        augment: bool = False,
        seed: int = 0,
        bank: ClipBank | None = None,
        simulate: SimulateFn | None = None,
    ) -> None:
        """`bank` turns on simulated occlusion (train only); `simulate` defaults to
        Dev A's occlusion.simulate_occlusion."""
        d = windows.load_windows(None, path=npz_path)
        self.kp, self.wh, self.y = d["kp"], d["wh"], d["y"]
        self.present, self.ctx = d["present"], d["ctx"]  # Contract 3 v2
        self.clip_id, self.end_frame = d["clip_id"], d["end_frame"]
        self.mu, self.sd, self.augment = stats["mu"], stats["sd"], augment
        self.rng = np.random.default_rng(seed)
        self.bank = bank
        self.simulate = simulate or (_dev_a_simulate() if bank is not None else None)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        kp, present, ctx = self.kp[i].copy(), self.present[i].copy(), self.ctx[i].copy()
        if self.augment:
            if self.bank is not None:
                occluded = maybe_occlude(
                    self.bank,
                    int(self.clip_id[i]),
                    int(self.end_frame[i]),
                    self.rng,
                    self.simulate,
                )
                if occluded is not None:
                    kp, present, ctx = occluded
            kp, present, ctx = augment_kp(kp, present, ctx, self.wh[i], self.rng)
        x = (make_features(kp, self.wh[i], present, ctx) - self.mu) / self.sd
        return torch.from_numpy(x.astype(np.float32)), int(self.y[i])


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own augmentation stream."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)
