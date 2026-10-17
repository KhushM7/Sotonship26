"""Rule baseline (E0): keypoint box wider than tall and hips near the floor for 2 s.

Per frame `low = present and log_bbox_aspect > a and hip_height < h` (features from
`make_features` over the whole clip). FALL fires once `low` has held 30 frames and
re-arms after 30 frames of not-low. Scores are written as Contract 4 rows every 3
frames (p_falling = 1 on the row where the rule fires, p_fallen = low), so the
evaluator scores the rule exactly like a model, with t_high = t_low = 0.5.
"""

from itertools import product

import numpy as np
import numpy.typing as npt
import pandas as pd

from fallsense import windows
from fallsense.config import load_config, skeleton_path
from fallsense.features import FEATURE_NAMES, make_features
from fallsense.occlusion import last_seen_context

CFG = load_config()
FPS = CFG["fps"]
GRID_A = (-0.2, 0.0, 0.2, 0.4)  # log(width / height) of the keypoint box
GRID_H = (0.3, 0.5, 0.7, 0.9)  # ankle-to-hip height in torso lengths
HOLD = 30  # frames low before FALL (2 s)
REARM = 30  # frames not-low before the rule can fire again
STEP = 3  # score rows every 3 frames, like the eval windows
THRESHOLDS = {"t_high": 0.5, "t_low": 0.5}
_ASPECT = FEATURE_NAMES.index("log_bbox_aspect")
_HIP = FEATURE_NAMES.index("hip_height")
_PRESENT = FEATURE_NAMES.index("detected")


def clip_signals(dataset: str, path: str) -> npt.NDArray[np.float32]:
    """(T, 3): log_bbox_aspect, hip_height, present for every frame of a clip."""
    d = np.load(skeleton_path(dataset, path))
    w, h = int(d["width"]), int(d["height"])
    kp = windows.select_keypoints(d["kp"], d["detected"], w, h)
    present = d["detected"].astype(bool)
    ctx = last_seen_context(kp, present, (w, h), FPS)
    x = make_features(kp, (w, h), present, ctx)
    return x[:, [_ASPECT, _HIP, _PRESENT]]


def rule(
    sig: npt.NDArray[np.float32], a: float, h: float
) -> tuple[npt.NDArray[np.bool_], list[int]]:
    """Per-frame `low` and the frames where FALL fires."""
    low = (sig[:, 2] > 0.5) & (sig[:, 0] > a) & (sig[:, 1] < h)
    fires, run_low, run_ok, armed = [], 0, 0, True
    for i, lo in enumerate(low):
        run_low, run_ok = (run_low + 1, 0) if lo else (0, run_ok + 1)
        if not armed and run_ok >= REARM:
            armed = True
        if armed and run_low >= HOLD:
            fires.append(i)
            armed = False
    return low, fires


def clip_scores(
    clip_id: int, low: npt.NDArray[np.bool_], fires: list[int]
) -> pd.DataFrame:
    """Contract 4 rows for one clip (ends every 3 frames from frame 29)."""
    end = np.arange(windows.WINDOW - 1, len(low), STEP)
    pf = np.zeros(len(end))
    for f in fires:
        k = np.searchsorted(end, f)  # first row at or after the firing frame
        if k < len(end):
            pf[k] = 1.0
    pfall = np.where(pf > 0, 0.0, low[end].astype(np.float64))
    return pd.DataFrame(
        {
            "clip_id": clip_id,
            "end_frame": end,
            "t_s": end / FPS,
            "p_upright": 1.0 - pf - pfall,
            "p_falling": pf,
            "p_fallen": pfall,
            "p_controlled_low": 0.0,
        }
    )


def scores_for(
    clips: pd.DataFrame, signals: dict[int, npt.NDArray[np.float32]], a: float, h: float
) -> pd.DataFrame:
    parts = [clip_scores(cid, *rule(signals[cid], a, h)) for cid in clips["clip_id"]]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def grid() -> list[tuple[float, float]]:
    return list(product(GRID_A, GRID_H))
