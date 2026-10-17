"""Event-level metrics for every system (model, baseline, tracker), README definitions.

- Fall event: each OmniFall `fall` segment (16-class label 1, after clean_segments).
- Detected: an alert between the segment start and 10 s after the segment end.
- False alert: an alert matching no fall event.
- Normal-activity hours: clip time outside [fall start - 2 s, fall end + 10 s].
- Latency: alert time minus fall segment start.

Alerts come from `calibrate.alerts_from_scores` (Dev B), never reimplemented here.
FALL and ON FLOOR are alarms; CHECK hands the decision to a person. Intervals are a
bootstrap over clips (windows of one clip are not independent).
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from fallsense.config import load_config
from fallsense.labels import clean_segments, segments_for

TEST_ONLY = tuple(load_config()["test_only_datasets"])

DETECT_AFTER_S = 10.0
NORMAL_BEFORE_S = 2.0
FALL_LABEL = 1  # OmniFall 16-class "fall"
ALARMS = ("FALL", "ON FLOOR")
CHECK = "CHECK"
N_BOOT = 1000
CLIP_COLUMNS = [
    "clip_id",
    "n_events",
    "det_alarm",
    "det_any",
    "check_only",
    "n_alarm",
    "n_check",
    "false_alarm",
    "false_check",
    "normal_h",
    "total_h",
]
# metric -> (numerator column, denominator column) for ratio metrics
RATIOS = {
    "sensitivity_alarm": ("det_alarm", "n_events"),
    "sensitivity_any": ("det_any", "n_events"),
    "false_alarms_per_h": ("false_alarm", "normal_h"),
    "checks_per_h": ("n_check", "total_h"),
}


def eval_clips(index: pd.DataFrame, split: str) -> pd.DataFrame:
    """Index rows evaluated for `split`, with clip_id = row number in index.csv.

    Same clips as 05_build_windows.py: train/val need qc_pass; test and
    staged_test keep every clip; `oops` is the test-only datasets.
    """
    df = index.reset_index(drop=True).rename_axis("clip_id").reset_index()
    test_only = df["dataset"].isin(TEST_ONLY)
    if split == "oops":
        return df[test_only].reset_index(drop=True)
    df = df[(df["split"] == split) & ~test_only]
    if split in ("train", "val"):
        df = df[df["qc_pass"].astype(bool)]
    return df.reset_index(drop=True)


def fall_events(clips: pd.DataFrame) -> pd.DataFrame:
    """Fall segments for `clips` (index rows with clip_id, dataset, path, duration_s).

    Returns DataFrame[clip_id, start, end] in seconds.
    """
    rows = []
    for r in clips.itertuples():
        segs = clean_segments(segments_for(r.dataset, r.path), float(r.duration_s))
        for s in segs[segs["label"] == FALL_LABEL].itertuples():
            rows.append((int(r.clip_id), float(s.start), float(s.end)))
    return pd.DataFrame(rows, columns=["clip_id", "start", "end"])


def _normal_seconds(duration: float, events: pd.DataFrame) -> float:
    """Clip time outside [start - 2 s, end + 10 s] of every event."""
    spans = sorted(
        (max(0.0, s - NORMAL_BEFORE_S), min(duration, e + DETECT_AFTER_S))
        for s, e in zip(events["start"], events["end"])
    )
    covered, cur_s, cur_e = 0.0, None, None
    for s, e in spans:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                covered += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        covered += cur_e - cur_s
    return max(0.0, duration - covered)


def match_events(
    clips: pd.DataFrame, events: pd.DataFrame, alerts: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-clip counts (CLIP_COLUMNS) and per-event rows with latencies.

    `clips`: clip_id, duration_s. `alerts`: clip_id, t_s, level (Contract 5 or
    alerts_from_scores output).
    """
    ev_by_clip = dict(iter(events.groupby("clip_id")))
    al_by_clip = dict(iter(alerts.groupby("clip_id")))
    empty_ev = events.iloc[:0]
    per_clip, per_event = [], []
    for cid, dur in zip(clips["clip_id"], clips["duration_s"]):
        ev = ev_by_clip.get(cid, empty_ev)
        al = al_by_clip.get(cid)
        t = al["t_s"].to_numpy(np.float64) if al is not None else np.empty(0)
        lv = al["level"].to_numpy() if al is not None else np.empty(0, object)
        alarm = np.isin(lv, ALARMS)
        check = lv == CHECK
        matched = np.zeros(len(t), bool)
        det_alarm = det_any = check_only = 0
        for s, e in zip(ev["start"], ev["end"]):
            hit = (t >= s) & (t <= e + DETECT_AFTER_S)
            matched |= hit
            first_alarm = t[hit & alarm].min() if (hit & alarm).any() else np.nan
            first_any = t[hit].min() if hit.any() else np.nan
            det_alarm += bool((hit & alarm).any())
            det_any += bool(hit.any())
            check_only += bool(hit.any() and not (hit & alarm).any())
            per_event.append((cid, s, e, first_alarm - s, first_any - s))
        per_clip.append(
            (
                cid,
                len(ev),
                det_alarm,
                det_any,
                check_only,
                int(alarm.sum()),
                int(check.sum()),
                int((alarm & ~matched).sum()),
                int((check & ~matched).sum()),
                _normal_seconds(float(dur), ev) / 3600,
                float(dur) / 3600,
            )
        )
    clip_df = pd.DataFrame(per_clip, columns=CLIP_COLUMNS)
    event_df = pd.DataFrame(
        per_event,
        columns=["clip_id", "start", "end", "latency_alarm", "latency_any"],
    )
    return clip_df, event_df


def summarise(clip_df: pd.DataFrame, event_df: pd.DataFrame) -> dict[str, float]:
    """Point estimates plus sample sizes."""
    tot = clip_df[CLIP_COLUMNS[1:]].sum()
    out: dict[str, float] = {
        name: _ratio(tot[num], tot[den]) for name, (num, den) in RATIOS.items()
    }
    out["coverage"] = _coverage(tot)
    out["median_latency_alarm_s"] = _nanmedian(event_df["latency_alarm"])
    out["median_latency_any_s"] = _nanmedian(event_df["latency_any"])
    out.update(
        events=int(tot["n_events"]),
        clips=len(clip_df),
        hours=float(tot["total_h"]),
        normal_hours=float(tot["normal_h"]),
        alarms=int(tot["n_alarm"]),
        checks=int(tot["n_check"]),
        false_alarms=int(tot["false_alarm"]),
    )
    return out


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def _nanmedian(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.median()) if len(x) else float("nan")


def _coverage(tot: pd.Series) -> float:
    """Share of alerts and true events decided without a CHECK."""
    items = tot["n_alarm"] + tot["n_check"] + tot["n_events"]
    handed_over = tot["n_check"] + tot["check_only"]
    return _ratio(items - handed_over, items)


def bootstrap(
    clip_df: pd.DataFrame,
    event_df: pd.DataFrame,
    n_boot: int = N_BOOT,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Percentile intervals, resampling clips with replacement."""
    rng = np.random.default_rng(seed)
    n = len(clip_df)
    if n == 0:
        return {}
    cols = clip_df[CLIP_COLUMNS[1:]].to_numpy(np.float64)
    pos = {cid: i for i, cid in enumerate(clip_df["clip_id"])}
    ev_clip = event_df["clip_id"].map(pos).to_numpy()
    lat = {k: event_df[f"latency_{k}"].to_numpy(np.float64) for k in ("alarm", "any")}
    stats: dict[str, list[float]] = {
        k: [] for k in [*RATIOS, "coverage", "median_latency_alarm_s"]
    }
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        tot = pd.Series(w @ cols, index=CLIP_COLUMNS[1:])
        for name, (num, den) in RATIOS.items():
            stats[name].append(_ratio(tot[num], tot[den]))
        stats["coverage"].append(_coverage(tot))
        reps = w[ev_clip].astype(int) if len(ev_clip) else np.empty(0, int)
        sample = np.repeat(lat["alarm"], reps)
        sample = sample[np.isfinite(sample)]
        stats["median_latency_alarm_s"].append(
            float(np.median(sample)) if len(sample) else np.nan
        )
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {
        k: (float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi)))
        for k, v in stats.items()
        if np.isfinite(v).any()
    }


def window_metrics(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, n_classes: int = 4
) -> dict[str, Any]:
    """4-class macro F1 and confusion matrix (rows true, columns predicted)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = np.zeros((n_classes, n_classes), int)
    np.add.at(cm, (y_true, y_pred), 1)
    f1 = []
    for c in range(n_classes):
        tp, fp, fn = cm[c, c], cm[:, c].sum() - cm[c, c], cm[c, :].sum() - cm[c, c]
        if tp + fp + fn:
            f1.append(2 * tp / (2 * tp + fp + fn))
    return {
        "macro_f1": float(np.mean(f1)) if f1 else float("nan"),
        "per_class_f1": f1,
        "confusion": cm.tolist(),
        "windows": len(y_true),
    }
