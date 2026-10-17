"""Tune the rule baseline (E0) on train clips and write its Contract 4 scores.

    python scripts/baseline_e0.py                      # tune on core train, score val
    python scripts/baseline_e0.py --splits val test oops

Grid search over (a, h) on **train** clips of the core staged sets, maximising event
F1 (recall = falls detected by FALL, precision = FALL alerts that match a fall).
Writes reports/scores/baseline/{split}.parquet, reports/scores/baseline/thresholds.yaml
and reports/scores/baseline/tuning.csv.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fallsense import baseline, metrics, scores
from fallsense.calibrate import alerts_from_scores
from fallsense.config import DATA, scores_path

CORE = ["GMDCSA24", "caucafall", "le2i", "mcfd"]


def _signals(args: tuple[int, str, str]) -> tuple[int, np.ndarray]:
    cid, dataset, path = args
    return cid, baseline.clip_signals(dataset, path)


def signals_for(clips: pd.DataFrame, workers: int) -> dict[int, np.ndarray]:
    jobs = list(zip(clips["clip_id"], clips["dataset"], clips["path"]))
    with ProcessPoolExecutor(workers) as ex:
        return dict(ex.map(_signals, jobs, chunksize=8))


def event_f1(clips: pd.DataFrame, events: pd.DataFrame, sc: pd.DataFrame) -> dict:
    alerts = alerts_from_scores(sc, baseline.THRESHOLDS)
    clip_df, event_df = metrics.match_events(clips, events, alerts)
    res = metrics.summarise(clip_df, event_df)
    alarms = res["alarms"]
    precision = (alarms - res["false_alarms"]) / alarms if alarms else 0.0
    recall = res["sensitivity_alarm"]
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, **res}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--index", type=Path, default=DATA / "skeletons" / "index.csv")
    ap.add_argument("--train-sets", nargs="+", default=CORE)
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    index = pd.read_csv(args.index)

    train = metrics.eval_clips(index, "train")
    train = train[train["dataset"].isin(args.train_sets)].reset_index(drop=True)
    events = metrics.fall_events(train)
    sig = signals_for(train, args.workers)
    print(f"tuning on {len(train)} train clips, {len(events)} falls")
    rows = []
    for a, h in baseline.grid():
        r = event_f1(train, events, baseline.scores_for(train, sig, a, h))
        rows.append({"a": a, "h": h, **r})
    tuning = pd.DataFrame(rows).sort_values(["f1", "recall"], ascending=False)
    best = tuning.iloc[0]
    a, h = float(best["a"]), float(best["h"])
    cols = ["a", "h", "f1", "precision", "recall", "false_alarms_per_h"]
    print(tuning[cols].round(3).to_string(index=False))
    print(f"chosen a={a}, h={h} (train event F1 {best['f1']:.3f})")

    out_dir = scores_path("baseline", "x").parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(out_dir / "tuning.csv", index=False)
    meta = {**baseline.THRESHOLDS, "a": a, "h": h, "train_sets": args.train_sets}
    with open(out_dir / "thresholds.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    for split in args.splits:
        clips = metrics.eval_clips(index, split)
        sc = baseline.scores_for(clips, signals_for(clips, args.workers), a, h)
        scores.save_scores("baseline", split, sc)
        print(f"{split}: {len(sc)} score rows for {len(clips)} clips")


if __name__ == "__main__":
    main()
