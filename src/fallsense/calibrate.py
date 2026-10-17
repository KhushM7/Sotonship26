"""Model scores (Contract 4) and the event logic that turns them into alerts.

`write_scores` runs a trained run (models/<run>/) over a Contract 3 windows file.
`alerts_from_scores` is the one implementation of FALL / CHECK / ON FLOOR used by
Dev A's evaluator, 07_calibrate.py and the live demo.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from fallsense import dataset, scores, windows
from fallsense.config import load_config
from fallsense.models import build_model

CFG = load_config()
FPS = CFG["fps"]
CLASS_COLUMNS = ["p_upright", "p_falling", "p_fallen", "p_controlled_low"]

FALL, CHECK, ON_FLOOR = "FALL", "CHECK", "ON FLOOR"
# a higher level may escalate through the refractory period of a lower one
LEVEL_RANK = {CHECK: 0, FALL: 1, ON_FLOOR: 2}
SCORE_WINDOWS = 3  # fall score = max p_falling over the last 3 eval windows (0.6 s)
REFRACTORY_S = 30.0
FALLEN_P = 0.5
ON_FLOOR_EVAL_S = 5.0  # public clips are short; the demo uses on_floor_seconds_demo
ALERT_COLUMNS = ["clip_id", "t_s", "level", "score"]


# ---------------------------------------------------------------- run files


def load_thresholds(model_dir: Path) -> dict[str, Any]:
    """models/<run>/thresholds.yaml, or {} before 07_calibrate.py has run."""
    path = Path(model_dir) / "thresholds.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_run(model_dir: Path) -> tuple[torch.nn.Module, dict[str, np.ndarray]]:
    """The trained model (eval mode) and its feature z-score stats."""
    model_dir = Path(model_dir)
    ckpt = torch.load(model_dir / "model.pt", map_location="cpu")
    model = build_model(ckpt["model"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with np.load(model_dir / "feat_stats.npz") as d:
        stats = {"mu": d["mu"], "sd": d["sd"]}
    return model, stats


@torch.no_grad()
def window_logits(
    model: torch.nn.Module,
    stats: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    batch: int = 1024,
) -> np.ndarray:
    """(N, 4) logits for every window, no augmentation."""
    x = ((dataset.window_features(arrays) - stats["mu"]) / stats["sd"]).astype(
        np.float32
    )
    out = [
        model(torch.from_numpy(x[i : i + batch])).numpy()
        for i in range(0, len(x), batch)
    ]
    return np.concatenate(out) if out else np.empty((0, windows.N_CLASSES), np.float32)


def softmax_t(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return torch.softmax(torch.from_numpy(logits) / temperature, dim=1).numpy()


def scores_frame(arrays: dict[str, np.ndarray], probs: np.ndarray) -> pd.DataFrame:
    """Contract 4 rows for windows `arrays` with class probabilities `probs`."""
    df = pd.DataFrame(
        {
            "clip_id": arrays["clip_id"],
            "end_frame": arrays["end_frame"],
            "t_s": arrays["end_frame"] / FPS,
        }
    )
    df[CLASS_COLUMNS] = probs
    return df.sort_values(["clip_id", "end_frame"], kind="stable").reset_index(
        drop=True
    )


def write_scores(
    model_dir: Path,
    split_npz: Path,
    out_parquet: Path | None = None,
    temperature: float | None = None,
) -> pd.DataFrame:
    """Score every window of `split_npz` with the run in `model_dir`.

    Temperature comes from the argument, else thresholds.yaml, else 1.0. Writes
    Contract 4 to `out_parquet` (default reports/scores/<run>/<split>.parquet)
    and returns the rows.
    """
    model_dir, split_npz = Path(model_dir), Path(split_npz)
    if temperature is None:
        temperature = float(load_thresholds(model_dir).get("temperature", 1.0))
    model, stats = load_run(model_dir)
    arrays = windows.load_windows(split_npz.stem, path=split_npz)
    probs = softmax_t(window_logits(model, stats, arrays), temperature)
    df = scores_frame(arrays, probs)
    scores.save_scores(model_dir.name, split_npz.stem, df, path=out_parquet)
    return df


# ---------------------------------------------------------------- events


def _clip_alerts(
    t: np.ndarray,
    s: np.ndarray,
    p_fallen: np.ndarray,
    t_high: float,
    t_low: float,
    on_floor_s: float,
) -> list[tuple[float, str, float]]:
    out: list[tuple[float, str, float]] = []
    last_t, last_rank = -np.inf, -1
    floor_start: float | None = None
    floor_done = False
    for ti, si, fi in zip(t, s, p_fallen):
        if fi > FALLEN_P:
            if floor_start is None:
                floor_start, floor_done = ti, False
        else:
            floor_start = None

        level, score = None, si
        if (
            floor_start is not None
            and not floor_done
            and ti - floor_start >= on_floor_s
        ):
            level, score, floor_done = ON_FLOOR, fi, True
        elif si >= t_high:
            level = FALL
        elif si >= t_low:
            level = CHECK
        if level is None:
            continue
        rank = LEVEL_RANK[level]
        if ti - last_t < REFRACTORY_S and rank <= last_rank:
            continue
        out.append((float(ti), level, float(score)))
        last_t, last_rank = ti, rank
    return out


def alerts_from_scores(
    scores_df: pd.DataFrame,
    thresholds: dict[str, Any],
    on_floor_seconds: float | None = None,
) -> pd.DataFrame:
    """Alerts per clip in time order: DataFrame[clip_id, t_s, level, score].

    - fall score s = p_falling, max over the last 3 evaluation windows (0.6 s)
    - s >= t_high -> FALL; t_low <= s < t_high -> CHECK
    - p_fallen > 0.5 continuously for `on_floor_seconds` -> ON FLOOR, once per run
      of fallen windows (default thresholds' on_floor_seconds_eval, else 5 s)
    - 30 s refractory after any alert; only a higher level (CHECK -> FALL ->
      ON FLOOR) may fire inside it, so a fall that first shows as CHECK still
      becomes FALL and a FALL still escalates to ON FLOOR
    `score` is s for FALL and CHECK, p_fallen for ON FLOOR.
    """
    t_high, t_low = float(thresholds["t_high"]), float(thresholds["t_low"])
    if on_floor_seconds is None:
        on_floor_seconds = float(
            thresholds.get("on_floor_seconds_eval", ON_FLOOR_EVAL_S)
        )
    rows = []
    df = scores_df.sort_values(["clip_id", "end_frame"], kind="stable")
    for clip_id, g in df.groupby("clip_id", sort=True):
        s = (
            g["p_falling"]
            .rolling(SCORE_WINDOWS, min_periods=1)
            .max()
            .to_numpy(np.float64)
        )
        for t_s, level, score in _clip_alerts(
            g["t_s"].to_numpy(np.float64),
            s,
            g["p_fallen"].to_numpy(np.float64),
            t_high,
            t_low,
            on_floor_seconds,
        ):
            rows.append((clip_id, t_s, level, score))
    out = pd.DataFrame(rows, columns=ALERT_COLUMNS)
    return out.astype(
        {"clip_id": np.int32, "t_s": np.float32, "level": str, "score": np.float32}
    )
