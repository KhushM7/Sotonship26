"""Occlusion handling (schema_version 2): last-seen context, disappearances, simulation.

`kp` arrays here are (T, 13, 3): pixel x, pixel y, visibility for the config
keypoints, NaN where no pose was found - the same array `features.py` takes.
"""

from collections import deque
from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from fallsense.config import load_config

_CFG = load_config()
OCC: dict[str, Any] = _CFG["occlusion"]
FPS: int = _CFG["fps"]
_KEYPOINTS: list[int] = list(_CFG["keypoints"])
HIPS = (_KEYPOINTS.index(23), _KEYPOINTS.index(24))
NOSE = _KEYPOINTS.index(0)
ANKLES = (_KEYPOINTS.index(27), _KEYPOINTS.index(28))

N_CTX = 4  # since_seen_s, last_hip_y, last_hip_vy, last_edge_dist
VEL_FRAMES = 5  # present frames used for the hip vertical speed
MIN_SEEN = 5  # present frames needed before a gap counts as a disappearance
MIN_GAP_S = 1.0
FILL_FRAMES = 2  # dropouts this short inside a sighting don't break it
NEVER_SEEN = (0.5, 0.0, 0.5)  # hip y, hip speed, edge distance before any sighting
BODY_WINDOW_S = 2.0  # body height = tallest the person stood in this window
MIN_BODY = 0.02  # body heights below this (/ frame height) are too small to trust
VISIBLE = 0.5  # a keypoint counts as visible above this visibility
GUESS_VIS = (0.0, 0.3)  # visibility MediaPipe gives keypoints it guesses
FLICKER_FRAMES = (3, 8)
FLICKER_P = 0.4
# `rise` edge position as a share of the hips' drop, measured up from their lowest
# point; tuned with scripts/occlusion_realism.py so simulated falls vanish mid-drop.
RISE_EDGE_IN_DROP = (0.3, 0.7)
MIN_DROP = 0.1  # hips dropping less than this share of body height = no fall

F32 = npt.NDArray[np.float32]
DISAPPEARANCE_COLUMNS = [
    "start",
    "end",
    "start_s",
    "end_s",
    "duration_s",
    "edge_dist",
    "descent_speed",
    "body_height",
    "returned",
    "kind",
]


def _height(kp_frame: npt.NDArray[Any], wh: npt.ArrayLike) -> float:
    """Person's height in this frame as a share of the frame height."""
    ys = kp_frame[:, 1]
    if not np.isfinite(ys).any():
        return 0.0
    return float(np.nanmax(ys) - np.nanmin(ys)) / float(np.asarray(wh)[1])


def _hip(kp_frame: npt.NDArray[Any], wh: npt.ArrayLike) -> tuple[float, float] | None:
    """Hip midpoint as (x / width, y / height), or None if the hips are missing."""
    xy = kp_frame[list(HIPS), :2]
    if not np.isfinite(xy).all():
        return None
    w, h = np.asarray(wh, dtype=np.float64)
    x, y = xy.mean(axis=0)
    return float(x / w), float(y / h)


class LastSeenState:
    """The four `ctx` values, updated one frame at a time (used live and offline)."""

    def __init__(self, fps: float = FPS, cap_s: float = OCC["ctx_cap_s"]) -> None:
        self.fps = fps
        self.cap_s = cap_s
        self.i = -1
        self.last_i: int | None = None
        self.hist: deque[tuple[int, float]] = deque(maxlen=VEL_FRAMES)
        self.heights: deque[tuple[int, float]] = deque()
        self.hip_y, self.vy, self.edge = NEVER_SEEN

    def update(
        self, kp_frame: npt.NDArray[Any], present: bool, wh: npt.ArrayLike
    ) -> F32:
        """Advance one frame; return [since_seen_s, hip_y, hip_vy, edge_dist]."""
        self.i += 1
        hip = _hip(kp_frame, wh) if present else None
        if hip is not None:
            x, y = hip
            self.hist.append((self.i, y))
            self.heights.append((self.i, _height(kp_frame, wh)))
            while self.i - self.heights[0][0] > BODY_WINDOW_S * self.fps:
                self.heights.popleft()
            body = max(h for _, h in self.heights)
            i0, y0 = self.hist[0]
            # Body heights/s, so the same fall reads the same near or far from the camera.
            if self.i > i0 and body > MIN_BODY:
                self.vy = (y - y0) * self.fps / (self.i - i0) / body
            else:
                self.vy = 0.0
            self.hip_y = y
            self.edge = float(np.clip(min(x, 1 - x, y, 1 - y), 0.0, 0.5))
            self.last_i = self.i
        if self.last_i is None:
            since = self.cap_s
        else:
            since = min((self.i - self.last_i) / self.fps, self.cap_s)
        return np.array([since, self.hip_y, self.vy, self.edge], np.float32)


def last_seen_context(
    kp: npt.NDArray[Any],
    present: npt.ArrayLike,
    wh: npt.ArrayLike,
    fps: float = FPS,
    cap_s: float = OCC["ctx_cap_s"],
) -> F32:
    """(T, 4) context per frame; compute over the whole clip, then slice windows.

    Columns: seconds since the pose was last present (0 while visible, capped at
    `cap_s`, `cap_s` if never seen); last visible hip height (y / H); last hip
    vertical speed in body heights/s over the last 5 present frames (+ = down; body
    height = the tallest the person stood in the last 2 s, so near and far cameras
    read the same);
    last hip distance to the nearest border (0-0.5).
    """
    present = np.asarray(present, dtype=bool)
    state = LastSeenState(fps, cap_s)
    out = np.empty((len(present), N_CTX), np.float32)
    for i in range(len(present)):
        out[i] = state.update(kp[i], bool(present[i]), wh)
    return out


def find_disappearances(
    kp: npt.NDArray[Any],
    present: npt.ArrayLike,
    wh: npt.ArrayLike,
    fps: float = FPS,
    cfg: dict[str, Any] = OCC,
) -> pd.DataFrame:
    """Every gap of at least 1 s after at least 5 present frames, classified.

    `body_height` is the person's tallest height (/ frame height) in the 2 s before
    the loss, so speed can also be judged in body heights/s.
    `kind` is `hidden_down` if the person was lost more than `edge_margin` from
    every border while their hips dropped faster than `descent_speed` (body
    heights/s), else `left`.
    `start`/`end` are frame indices (`end` exclusive; the clip length if they never
    came back). Gaps are found on `stable_presence`, so a hidden person flickering
    back for a frame or two stays one disappearance.
    """
    present = np.asarray(present, dtype=bool)
    ctx = last_seen_context(kp, present, wh, fps, cfg["ctx_cap_s"])
    stable = stable_presence(present)
    min_gap = round(MIN_GAP_S * fps)
    rows = []
    i, n = 0, len(stable)
    while i < n:
        if stable[i]:
            i += 1
            continue
        j = i
        while j < n and not stable[j]:
            j += 1
        if j - i >= min_gap and i >= MIN_SEEN and stable[i - MIN_SEEN : i].all():
            _, _, vy, edge = ctx[i]  # carried over from the last present frame
            body = _body_height(kp, present, i, wh, fps)
            hidden = edge > cfg["edge_margin"] and vy > cfg["descent_speed"]
            rows.append(
                {
                    "start": i,
                    "end": j,
                    "start_s": i / fps,
                    "end_s": j / fps,
                    "duration_s": (j - i) / fps,
                    "edge_dist": float(edge),
                    "descent_speed": float(vy),
                    "body_height": body,
                    "returned": j < n,
                    "kind": "hidden_down" if hidden else "left",
                }
            )
        i = j
    return pd.DataFrame(rows, columns=DISAPPEARANCE_COLUMNS)


def _runs(x: npt.NDArray[np.bool_]) -> list[tuple[int, int, bool]]:
    """(start, end exclusive, value) for each run of equal values."""
    edges = np.flatnonzero(np.diff(x.astype(np.int8))) + 1
    bounds = [0, *edges.tolist(), len(x)]
    return [(s, e, bool(x[s])) for s, e in pairwise(bounds) if e > s]


def stable_presence(
    present: npt.ArrayLike, fill: int = FILL_FRAMES, min_run: int = MIN_SEEN
) -> npt.NDArray[np.bool_]:
    """Presence with dropouts of up to `fill` frames inside a sighting filled in,
    then sightings shorter than `min_run` frames removed (detector blips)."""
    p = np.asarray(present, dtype=bool).copy()
    for s, e, v in _runs(p):
        if not v and 0 < s and e < len(p) and e - s <= fill:
            p[s:e] = True
    for s, e, v in _runs(p):
        if v and e - s < min_run:
            p[s:e] = False
    return p


def _body_height(
    kp: npt.NDArray[Any],
    present: npt.NDArray[np.bool_],
    i: int,
    wh: npt.ArrayLike,
    fps: float,
) -> float:
    """Tallest the person stood (/ frame height) in the 2 s before frame `i`."""
    lo = max(0, i - round(2 * fps))
    ys = kp[lo:i][present[lo:i], :, 1]
    if not len(ys):
        return float("nan")
    heights = np.nanmax(ys, axis=1) - np.nanmin(ys, axis=1)
    return float(np.nanmax(heights)) / float(np.asarray(wh)[1])


def simulate_occlusion(
    kp: npt.NDArray[Any],
    present: npt.ArrayLike,
    wh: npt.ArrayLike,
    rng: np.random.Generator,
    onset: int,
    mode: str = "rise",
    cfg: dict[str, Any] = OCC,
) -> tuple[F32, npt.NDArray[np.bool_]]:
    """Hide the person from frame `onset` on, as furniture would.

    `rise`: an occluder edge at a fixed image height just above the person's lowest
    posture after onset; keypoints that go below it are hidden for good, so ankles
    vanish first and the head last, as when sinking behind a bed (someone who never
    goes down vanishes at once, as when walking behind a tall cupboard). `box`: a fixed rectangle over
    20-50% of the person's box; keypoints inside it are hidden while they are there.
    Hidden keypoints become low-visibility guesses that still follow the body (what
    MediaPipe does with hidden limbs), so the hip motion before the loss is kept;
    once both hips are hidden, or fewer than `min_visible_kp` keypoints are
    visible, the whole frame is lost (NaN, not present). The 3-8 frames before the loss flicker.
    """
    kp_out = np.array(kp, dtype=np.float32, copy=True)
    present_out = np.array(present, dtype=bool, copy=True)
    n = len(present_out)
    ref = _reference_frame(kp_out, present_out, onset)
    if ref is None or not present_out[onset:].any():
        return kp_out, present_out  # nobody visible from onset on: nothing to hide

    xy = kp_out[ref, :, :2]
    ok = np.isfinite(xy).all(axis=1)
    (x0, y0), (x1, y1) = xy[ok].min(axis=0), xy[ok].max(axis=0)
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    frames = np.arange(onset, n)
    ys, xs = kp_out[onset:, :, 1].copy(), kp_out[onset:, :, 0].copy()

    if mode == "rise":
        # The person vanishes when the hips cross the furniture's top edge. Put the
        # edge part-way down the hips' drop, where a fall is fastest; with no drop
        # (walking behind a sofa) the edge sits at hip height and they vanish at once.
        hip_y = np.nanmean(kp_out[:, list(HIPS), 1], axis=1)
        hip_start = float(hip_y[ref])
        hip_low = float(hip_y[_lowest_frame(kp_out, present_out, onset)])
        drop = hip_low - hip_start
        if drop > MIN_DROP * bh:
            edge_y = hip_low - rng.uniform(*RISE_EDGE_IN_DROP) * drop
        else:
            edge_y = hip_start - rng.uniform(0.0, 0.1) * bh
        below = np.nan_to_num(ys, nan=-np.inf) > edge_y
        hidden = np.logical_or.accumulate(below, axis=0)  # once hidden, stays hidden
    elif mode == "box":
        frac = rng.uniform(0.2, 0.5)
        box_w = bw * rng.uniform(np.sqrt(frac), 1.0)
        box_h = min(bh, frac * bw * bh / box_w)
        bx = x0 + rng.uniform(0, bw - box_w)
        by = y0 + rng.uniform(0, bh - box_h)
        hidden = (xs >= bx) & (xs <= bx + box_w) & (ys >= by) & (ys <= by + box_h)
    else:
        raise ValueError(f"unknown occlusion mode {mode!r}")

    seg = kp_out[onset:]
    hidden &= present_out[onset:, None]
    seg[..., 0] = np.where(hidden, xs + rng.normal(0, 0.02 * bw, xs.shape), xs)
    seg[..., 1] = np.where(hidden, ys + rng.normal(0, 0.02 * bh, ys.shape), ys)
    seg[..., 2] = np.where(hidden, rng.uniform(*GUESS_VIS, size=xs.shape), seg[..., 2])
    visible = (~hidden & (seg[..., 2] > VISIBLE)).sum(axis=1)
    # MediaPipe drops the person as soon as both hips are hidden (OCCU: hips stay
    # visible right up to the loss), or when too few keypoints are left.
    hips_hidden = hidden[:, list(HIPS)].all(axis=1)
    lost = present_out[onset:] & (hips_hidden | (visible < cfg["min_visible_kp"]))

    if lost.any():
        first = int(frames[lost.argmax()])
        k = int(rng.integers(FLICKER_FRAMES[0], FLICKER_FRAMES[1] + 1))
        for t in range(max(onset, first - k), first):
            if present_out[t] and rng.random() < FLICKER_P:
                lost[t - onset] = True
    kp_out[onset:][lost] = np.nan
    present_out[onset:][lost] = False
    return kp_out, present_out


def _lowest_frame(
    kp: npt.NDArray[Any], present: npt.NDArray[np.bool_], onset: int
) -> int:
    """Present frame from `onset` on where the hips are lowest in the image."""
    frames = onset + np.flatnonzero(present[onset:])
    hip_y = np.nanmean(kp[frames][:, list(HIPS), 1], axis=1)
    return int(frames[np.nanargmax(hip_y)])


def _reference_frame(
    kp: npt.NDArray[Any], present: npt.NDArray[np.bool_], onset: int
) -> int | None:
    """First present frame from `onset` on, else the last one before it."""
    after = np.flatnonzero(present[onset:])
    if len(after):
        return onset + int(after[0])
    before = np.flatnonzero(present[:onset])
    return int(before[-1]) if len(before) else None
