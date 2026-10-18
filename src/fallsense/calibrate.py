"""Model scores (Contract 4) and the event logic that turns them into alerts.

`write_scores` runs a trained run (models/<run>/) over a Contract 3 windows file.
`alerts_from_scores` is the one implementation of FALL / CHECK / ON FLOOR used by
Dev A's evaluator, 07_calibrate.py and the live demo.
"""

from collections import deque
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


def ensemble_members(model_dir: Path) -> list[Path]:
    """Member run folders listed in models/<run>/ensemble.yaml, or [] for a plain run."""
    path = Path(model_dir) / "ensemble.yaml"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        names = yaml.safe_load(f)["members"]
    return [Path(model_dir).parent / n for n in names]


def ensemble_logits(member_logits: list[np.ndarray]) -> np.ndarray:
    """log of the mean member softmax: an ensemble's logits (softmax gives the mean)."""
    probs = np.mean([softmax_t(lg) for lg in member_logits], axis=0)
    return np.log(np.maximum(probs, 1e-12)).astype(np.float32)


class Scorer:
    """Window logits from raw (N, 30, 51) features, for one run or an ensemble.

    An ensemble folder holds ensemble.yaml (members: [run, ...]) and averages its
    members' probabilities, each member with its own feat_stats.npz.
    """

    def __init__(self, model_dir: Path) -> None:
        members = ensemble_members(model_dir)
        self.ensemble = bool(members)
        self.runs = [load_run(m) for m in members or [Path(model_dir)]]

    @torch.no_grad()
    def logits(self, feats: np.ndarray, batch: int = 1024) -> np.ndarray:
        out = []
        for model, stats in self.runs:
            x = ((feats - stats["mu"]) / stats["sd"]).astype(np.float32)
            parts = [
                model(torch.from_numpy(x[i : i + batch])).numpy()
                for i in range(0, len(x), batch)
            ]
            out.append(
                np.concatenate(parts)
                if parts
                else np.empty((0, windows.N_CLASSES), np.float32)
            )
        return ensemble_logits(out) if self.ensemble else out[0]


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
    arrays = windows.load_windows(split_npz.stem, path=split_npz)
    logits = Scorer(model_dir).logits(dataset.window_features(arrays))
    probs = softmax_t(logits, temperature)
    df = scores_frame(arrays, probs)
    scores.save_scores(model_dir.name, split_npz.stem, df, path=out_parquet)
    return df


# ---------------------------------------------------------------- events


class AlertState:
    """The event logic for one clip (or one live stream), one evaluation window at a time.

    `step` takes the window's p_falling and p_fallen and returns (level, score) when
    an alert fires; the fall score is the max p_falling over the last
    SCORE_WINDOWS windows.
    """

    def __init__(self, t_high: float, t_low: float, on_floor_s: float) -> None:
        self.t_high, self.t_low, self.on_floor_s = t_high, t_low, on_floor_s
        self.recent: deque[float] = deque(maxlen=SCORE_WINDOWS)
        self.last_t, self.last_rank = -np.inf, -1
        self.floor_start: float | None = None
        self.floor_done = False

    def step(
        self, t: float, p_falling: float, p_fallen: float
    ) -> tuple[str, float] | None:
        self.recent.append(p_falling)
        s = max(self.recent)
        if p_fallen > FALLEN_P:
            if self.floor_start is None:
                self.floor_start, self.floor_done = t, False
        else:
            self.floor_start = None

        level, score = None, s
        if (
            self.floor_start is not None
            and not self.floor_done
            and t - self.floor_start >= self.on_floor_s
        ):
            level, score, self.floor_done = ON_FLOOR, p_fallen, True
        elif s >= self.t_high:
            level = FALL
        elif s >= self.t_low:
            level = CHECK
        if level is None:
            return None
        rank = LEVEL_RANK[level]
        if t - self.last_t < REFRACTORY_S and rank <= self.last_rank:
            return None
        self.last_t, self.last_rank = t, rank
        return level, float(score)


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
        state = AlertState(t_high, t_low, on_floor_seconds)
        for t_s, p_falling, p_fallen in zip(
            g["t_s"].to_numpy(np.float64),
            g["p_falling"].to_numpy(np.float64),
            g["p_fallen"].to_numpy(np.float64),
        ):
            hit = state.step(float(t_s), float(p_falling), float(p_fallen))
            if hit is not None:
                rows.append((clip_id, float(t_s), hit[0], hit[1]))
    out = pd.DataFrame(rows, columns=ALERT_COLUMNS)
    return out.astype(
        {"clip_id": np.int32, "t_s": np.float32, "level": str, "score": np.float32}
    )
