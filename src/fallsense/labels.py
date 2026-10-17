"""OmniFall label loading and the 16-class -> 4-class mapping."""

from collections import Counter
from functools import cache

import numpy as np
import pandas as pd

from fallsense.config import load_config

_CFG = load_config()
TEST_ONLY = set(_CFG["test_only_datasets"])

# OmniFall: 0 walk, 1 fall, 2 fallen, 3 sit_down, 4 sitting, 5 lie_down, 6 lying,
# 7 stand_up, 8 standing, 9 other, 10-15 extended (kneel, squat, crawl, jump).
# Ours:     0 upright, 1 falling, 2 fallen, 3 controlled_low.
TO4 = np.array([0, 1, 2, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
UNLABELLED = -1
OVERLAP_TOL_S = 1e-3


def config_for(dataset: str) -> str:
    """OmniFall config holding a dataset's labels (cross-subject split)."""
    if dataset == "of-syn":
        return "of-syn"
    if dataset == "OOPS":
        return "of-itw"
    return f"{dataset.lower()}{_CFG['split_config_suffix']}"


@cache
def load_labels(dataset: str) -> pd.DataFrame:
    """Every OmniFall segment for one dataset, all splits, with a `split` column.

    `validation` is renamed `val`; test-only datasets (OOPS) put every row in `test`.
    """
    import omnifall

    dd = omnifall.load(config_for(dataset))
    parts = []
    for split, ds in dd.items():
        df = ds.to_pandas()
        df["split"] = "val" if split == "validation" else split
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df = df[df["dataset"] == dataset].reset_index(drop=True)
    if dataset in TEST_ONLY:
        df["split"] = "test"
    return df


@cache
def _segments_by_path(dataset: str) -> dict[str, pd.DataFrame]:
    return {
        path: g.reset_index(drop=True)
        for path, g in load_labels(dataset).groupby("path")
    }


def segments_for(dataset: str, path: str) -> pd.DataFrame:
    """All OmniFall segments for one clip (empty frame if the clip has none)."""
    segs = _segments_by_path(dataset).get(path)
    if segs is None:
        return load_labels(dataset).iloc[0:0]
    return segs.copy()


def clean_segments(
    segs: pd.DataFrame, duration_s: float, counts: Counter | None = None
) -> pd.DataFrame:
    """Clip segments to [0, duration_s], drop empty ones, sort by start.

    Overlaps are left in place and resolved by `frame_labels` (later start wins).
    If `counts` is given, it is updated with neg_start_clipped, end_clipped,
    zero_length_dropped and overlaps.
    """
    s = segs.copy()
    if counts is not None:
        counts["segments"] += len(s)
        counts["neg_start_clipped"] += int((s["start"] < 0).sum())
        counts["end_clipped"] += int((s["end"] > duration_s).sum())
    s["start"] = s["start"].clip(lower=0)
    s["end"] = s["end"].clip(upper=duration_s)
    keep = s["end"] > s["start"]
    s = s[keep].sort_values("start", kind="stable").reset_index(drop=True)
    if counts is not None:
        counts["zero_length_dropped"] += int((~keep).sum())
        # Touching segments differ by float32 rounding; only count real overlaps.
        prev_end = s["end"].cummax().to_numpy()[:-1]
        counts["overlaps"] += int(
            (s["start"].to_numpy()[1:] < prev_end - OVERLAP_TOL_S).sum()
        )
    return s


def frame_labels(segs: pd.DataFrame, t_ms: np.ndarray) -> np.ndarray:
    """16-class id per kept frame; later segments win on overlap; gaps = -1."""
    t = np.asarray(t_ms) / 1000.0
    y = np.full(len(t), UNLABELLED, dtype=np.int8)
    for start, end, lab in segs[["start", "end", "label"]].itertuples(index=False):
        y[(t >= start) & (t < end)] = lab
    return y


def to4(y16: np.ndarray) -> np.ndarray:
    """Map OmniFall ids 0-15 to our 4 classes; -1 stays -1."""
    out = np.full_like(y16, UNLABELLED)
    m = y16 >= 0
    out[m] = TO4[y16[m]]
    return out
