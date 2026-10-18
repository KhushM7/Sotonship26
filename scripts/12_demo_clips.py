"""Suggest clips for the demo rooms from a run's stored scores (no model run needed).

    python scripts/12_demo_clips.py --run e5_gru_s1 --split val

Prints two lists for configs/rooms.yaml:
- quiet clips: no alert, low fall score, person seen throughout (background rooms)
- fall clips: a FALL alert, ending low; with --hold seconds of the last pose kept,
  whether ON FLOOR also fires (the demo's on_floor_seconds_demo)
"""

import argparse

import numpy as np
import pandas as pd

from fallsense.calibrate import alerts_from_scores
from fallsense_server.clips import ClipRef, _index, _scores, load_clip, load_thresholds

CORE = ["GMDCSA24", "caucafall", "le2i", "mcfd", "occu"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="e5_gru_s1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--hold", type=float, default=40.0)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--quiet-max", type=float, default=0.15)
    args = ap.parse_args()

    idx = _index()
    sc = _scores(args.run, args.split)
    th = load_thresholds(args.run)
    alerts = alerts_from_scores(sc, th, on_floor_seconds=5.0)
    n_alerts = alerts.groupby("clip_id").size()
    per_clip = sc.groupby("clip_id").agg(
        max_falling=("p_falling", "max"), end_fallen=("p_fallen", "last")
    )
    info = idx.loc[per_clip.index, ["dataset", "path", "duration_s", "det_rate"]]
    per_clip = per_clip.join(info)
    per_clip["alerts"] = n_alerts.reindex(per_clip.index).fillna(0).astype(int)
    per_clip = per_clip[per_clip["dataset"].isin(CORE)]

    quiet = per_clip[
        (per_clip["alerts"] == 0)
        & (per_clip["max_falling"] < args.quiet_max)
        & (per_clip["det_rate"] > 0.97)
    ].sort_values("duration_s", ascending=False)
    print(f"Quiet clips ({len(quiet)}):")
    _print(quiet.head(args.top))

    falls = per_clip[(per_clip["alerts"] > 0) & (per_clip["end_fallen"] > 0.6)]
    rows = []
    for cid, r in falls.iterrows():
        clip = load_clip(ClipRef(r.dataset, r.path, hold_s=args.hold), args.run)
        levels = [a[1] for a in clip.alerts]
        first = next((a[0] for a in clip.alerts if a[1] == "FALL"), np.nan)
        rows.append({"clip_id": cid, "first_fall_s": first, "levels": " ".join(levels)})
    out = falls.join(pd.DataFrame(rows).set_index("clip_id"))
    out = out[out["levels"].str.contains("ON FLOOR")].sort_values("first_fall_s")
    print(f"\nFall clips that reach ON FLOOR with --hold {args.hold:g} ({len(out)}):")
    _print(out.head(args.top), extra=["first_fall_s", "levels"])


def _print(df: pd.DataFrame, extra: list[str] | None = None) -> None:
    cols = ["dataset", "path", "duration_s", "max_falling", *(extra or [])]
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(df[cols].round(2).to_string())


if __name__ == "__main__":
    main()
