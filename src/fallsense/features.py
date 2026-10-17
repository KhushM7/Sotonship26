"""Per-frame features from 13-keypoint pixel skeletons (51 per frame).

Used by training, evaluation, the rule baseline and the live demo, so there is
no train/serve mismatch. Everything here is causal (uses only past frames), so
the live ring buffer gives the same numbers as offline windows.
"""

import numpy as np
from numpy.typing import ArrayLike

from fallsense.config import load_config

_CFG = load_config()
FPS = _CFG["fps"]
CTX_CAP_S = _CFG["occlusion"]["ctx_cap_s"]

KP_NAMES = [
    "nose",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
]
# positions within the 13 config keypoints
NOSE, L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 0, 1, 2, 7, 8, 11, 12

FILL_LIMIT = 5  # frames (0.33 s) a lost keypoint is carried forward
VIS_OK = 0.5
CLIP = 10.0

DERIVED_NAMES = [
    "torso_angle",
    "hip_height",
    "head_height",
    "hip_vel_y",
    "hip_speed",
    "log_bbox_aspect",
    "mean_vis",
    "detected",
]
CTX_NAMES = [
    "ctx_since_seen",
    "ctx_last_hip_y",
    "ctx_last_hip_vy",
    "ctx_last_edge_dist",
]
FEATURE_NAMES = (
    [f"{n}_{c}" for n in KP_NAMES for c in ("x", "y")]
    + [f"{n}_vis" for n in KP_NAMES]
    + DERIVED_NAMES
    + CTX_NAMES
)
N_FEATURES = len(FEATURE_NAMES)  # 51


def _ffill(kp: np.ndarray, limit: int) -> np.ndarray:
    """Carry each keypoint's last valid (x, y, vis) forward over at most `limit` NaN frames."""
    # causal (pandas-style ffill with a limit), so live and offline agree
    out = kp.copy()
    for j in range(kp.shape[1]):
        last, run = None, 0
        for t in range(kp.shape[0]):
            if np.isfinite(kp[t, j, :2]).all():
                last, run = kp[t, j], 0
            elif last is not None and run < limit:
                out[t, j] = last
                run += 1
    return out


def _mid(xy: np.ndarray, a: int, b: int) -> np.ndarray:
    return (xy[:, a] + xy[:, b]) / 2


def torso_length(xy: np.ndarray, vis: np.ndarray, height: float) -> float:
    """Median shoulder-mid to hip-mid distance over well-seen frames, in pixels."""
    ok = (vis[:, [L_SH, R_SH, L_HIP, R_HIP]] > VIS_OK).all(axis=1)
    d = np.linalg.norm(_mid(xy, L_SH, R_SH) - _mid(xy, L_HIP, R_HIP), axis=1)
    ok &= np.isfinite(d)
    L = float(np.median(d[ok])) if ok.any() else 0.25 * height
    return max(L, 0.05 * height)


def make_features(
    kp: ArrayLike,
    wh: ArrayLike,
    present: ArrayLike,
    ctx: ArrayLike,
    fps: float = FPS,
    ctx_cap_s: float = CTX_CAP_S,
) -> np.ndarray:
    """(T, 13, 3) pixel x, y, visibility (NaN where lost) -> (T, 51) float32.

    `wh` is (width, height); `present` (T,) bool; `ctx` (T, 4) from
    occlusion.last_seen_context.
    """
    kp = _ffill(np.asarray(kp, dtype=np.float64), FILL_LIMIT)
    present = np.asarray(present, dtype=bool)
    ctx = np.asarray(ctx, dtype=np.float64)
    T = kp.shape[0]
    height = float(np.asarray(wh)[1])

    xy = kp[..., :2]  # NaN where still missing
    lost = ~np.isfinite(xy).all(axis=-1)
    vis = np.where(lost, 0.0, np.nan_to_num(kp[..., 2]))

    L = torso_length(xy, vis, height)
    hip = _mid(xy, L_HIP, R_HIP)
    sh = _mid(xy, L_SH, R_SH)
    ank = _mid(xy, L_ANK, R_ANK)

    shape = ((xy - hip[:, None]) / L).reshape(T, -1)

    v = sh - hip
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_up = -v[:, 1] / np.linalg.norm(v, axis=1)  # image up is -y
    torso_angle = np.arccos(np.clip(cos_up, -1, 1)) / np.pi
    hip_height = (ank[:, 1] - hip[:, 1]) / L
    head_height = (hip[:, 1] - xy[:, NOSE, 1]) / L

    d_hip = np.diff(hip, axis=0, prepend=hip[:1]) * fps / L
    hip_vel_y = d_hip[:, 1]
    hip_speed = np.linalg.norm(d_hip, axis=1)

    seen = (vis > VIS_OK) & ~lost
    enough = seen.sum(axis=1) >= 2
    x_seen, y_seen = xy[..., 0][enough], xy[..., 1][enough]
    s = seen[enough]
    bw = np.where(s, x_seen, -np.inf).max(1) - np.where(s, x_seen, np.inf).min(1)
    bh = np.where(s, y_seen, -np.inf).max(1) - np.where(s, y_seen, np.inf).min(1)
    log_aspect = np.full(T, np.nan)
    log_aspect[enough] = np.log(np.maximum(bw, 1.0) / np.maximum(bh, 1.0))

    derived = np.stack(
        [
            torso_angle,
            hip_height,
            head_height,
            hip_vel_y,
            hip_speed,
            log_aspect,
            vis.mean(axis=1),
            present.astype(np.float64),
        ],
        axis=1,
    )
    context = ctx.copy()
    context[:, 0] = ctx[:, 0] / ctx_cap_s

    x = np.concatenate([shape, vis, derived, context], axis=1)
    x = np.clip(np.nan_to_num(x, nan=0.0), -CLIP, CLIP).astype(np.float32)
    assert x.shape == (T, N_FEATURES) and np.isfinite(x).all()
    return x
