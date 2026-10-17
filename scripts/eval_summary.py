"""Compare every evaluated run on one split, per run and per experiment (mean over seeds).

    python scripts/eval_summary.py                 # val, every run with a report
    python scripts/eval_summary.py --runs e1_gru_s1 e2_gru_s1

Reads reports/eval/eval_<run>_<split>.json (from 08_evaluate.py) and the run's
scores. Adds two threshold-free numbers, because seeds land at different points of
the sensitivity / false-alert trade-off at a fixed threshold:
- window AP for falling-vs-rest (the training selection metric);
- the best operating point: lowest false FALL alerts/h among FALL thresholds that
  keep FALL sensitivity >= --target (sweep 0.05-0.95).
Writes reports/eval/summary_<split>.md and .csv.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fallsense import metrics, scores, windows
from fallsense.calibrate import alerts_from_scores
from fallsense.config import DATA, windows_path

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "reports" / "eval"
CORE_VAL = ["GMDCSA24", "caucafall", "le2i"]  # MCFD has no val clips
SWEEP = np.round(np.arange(0.05, 0.96, 0.05), 2)


def experiment(run: str) -> str:
    return re.sub(r"_s\d+$", "", run)


def group(report: dict, name: str, value: str) -> dict:
    for b in report["breakdown"]:
        if b["group"] == name and b["value"] == value:
            return b
    return {}


def best_point(
    sc: pd.DataFrame, clips, events, target: float
) -> tuple[float, float, float]:
    """(t_high, sensitivity, false/h) with the fewest false alerts at sens >= target."""
    best = (np.nan, np.nan, np.inf)
    for t in SWEEP:
        res = metrics.summarise(
            *metrics.match_events(
                clips, events, alerts_from_scores(sc, {"t_high": t, "t_low": t})
            )
        )
        if res["sensitivity_alarm"] >= target and res["false_alarms_per_h"] < best[2]:
            best = (float(t), res["sensitivity_alarm"], res["false_alarms_per_h"])
    return best if np.isfinite(best[2]) else (np.nan, np.nan, np.nan)


def core_events(clips: pd.DataFrame, events: pd.DataFrame, sc: pd.DataFrame):
    c = clips[clips["dataset"].isin(CORE_VAL)]
    ids = set(c["clip_id"])
    return c, events[events["clip_id"].isin(ids)], sc[sc["clip_id"].isin(ids)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="val")
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--index", type=Path, default=DATA / "skeletons" / "index.csv")
    args = ap.parse_args()

    runs = args.runs or sorted(
        p.name[len("eval_") : -len(f"_{args.split}.json")]
        for p in EVAL.glob(f"eval_*_{args.split}.json")
    )
    index = pd.read_csv(args.index)
    clips = metrics.eval_clips(index, args.split)
    events = metrics.fall_events(clips)
    w = windows.load_windows(args.split, path=windows_path(args.split))
    key = pd.DataFrame(
        {"clip_id": w["clip_id"], "end_frame": w["end_frame"], "y": w["y"]}
    )

    rows = []
    for run in runs:
        rep = json.loads((EVAL / f"eval_{run}_{args.split}.json").read_text("utf-8"))
        main_res = rep["systems"][rep["breakdown_system"]]
        sc = scores.load_scores(run, args.split)
        sc = sc[sc["clip_id"].isin(set(clips["clip_id"]))]
        m = key.merge(sc, on=["clip_id", "end_frame"])
        window_ap = average_precision_score(m["y"] == windows.FALLING, m["p_falling"])
        t, s, f = best_point(sc, clips, events, args.target)
        cc, ce, cs = core_events(clips, events, sc)
        core = metrics.summarise(
            *metrics.match_events(cc, ce, alerts_from_scores(cs, rep["thresholds"]))
        )
        rows.append(
            {
                "run": run,
                "experiment": experiment(run),
                "t_high": rep["thresholds"]["t_high"],
                "window_ap_falling": window_ap,
                "window_macro_f1": rep["windows"].get("macro_f1", np.nan),
                "sens_fall": main_res["sensitivity_alarm"],
                "false_per_h": main_res["false_alarms_per_h"],
                "latency_s": main_res["median_latency_alarm_s"],
                "core_sens_fall": core["sensitivity_alarm"],
                "core_false_per_h": core["false_alarms_per_h"],
                "ofsyn_sens_fall": group(rep, "dataset", "of-syn").get(
                    "sensitivity_alarm", np.nan
                ),
                "elderly_sens_fall": group(rep, "of-syn age", "elderly").get(
                    "sensitivity_alarm", np.nan
                ),
                "occu_sens_fall": group(rep, "dataset", "occu").get(
                    "sensitivity_alarm", np.nan
                ),
                "best_t": t,
                "best_sens": s,
                "best_false_per_h": f,
            }
        )
        print(
            f"{run:14s} AP {window_ap:.3f}  sens {main_res['sensitivity_alarm']:.3f}  best t {t}"
        )
    per_run = pd.DataFrame(rows)
    num = per_run.columns.drop(["run", "experiment"])
    per_exp = per_run.groupby("experiment")[num].agg(["mean", "min", "max"])
    seeds = per_run.groupby("experiment").size().rename("seeds")
    EVAL.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(EVAL / f"summary_{args.split}.csv", index=False)
    write_md(per_run, per_exp, seeds, args, rows[0] if rows else {}, clips, events)


def pct(x: float) -> str:
    return "–" if pd.isna(x) else f"{100 * x:.1f}%"


def num(x: float) -> str:
    return "–" if pd.isna(x) else f"{x:.2f}"


def write_md(per_run, per_exp, seeds, args, _first, clips, events) -> None:
    core = clips[clips["dataset"].isin(CORE_VAL)]
    lines = [
        f"# Run comparison on {args.split}",
        "",
        (
            f"{len(clips)} clips, {len(events)} falls ({len(core)} clips / "
            f"{int(events['clip_id'].isin(set(core['clip_id'])).sum())} falls in the core sets "
            f"{', '.join(CORE_VAL)}). Event numbers use each run's thresholds.yaml; "
            "the best operating point sweeps the FALL threshold instead. Per-run reports: "
            "`eval_<run>_<split>.md`."
        ),
        "",
        "## Per experiment (mean over seeds, min–max in brackets)",
        "",
        (
            "| Experiment | Seeds | Window AP (falling) | Macro F1 | Sens. FALL | False FALL / h "
            "| Core sens. | Core false / h | OF-Syn sens. | Elderly sens. | OCCU sens. "
            f"| Fewest false / h at ≥ {args.target:.0%} sens. |"
        ),
        "| --- " * 12 + "|",
    ]

    def cell(exp, col, f):
        r = per_exp.loc[exp, col]
        return f"{f(r['mean'])} ({f(r['min'])}–{f(r['max'])})"

    for exp in per_exp.index:
        lines.append(
            f"| {exp} | {seeds[exp]} | {cell(exp, 'window_ap_falling', num)} | "
            f"{cell(exp, 'window_macro_f1', num)} | {cell(exp, 'sens_fall', pct)} | "
            f"{cell(exp, 'false_per_h', num)} | {cell(exp, 'core_sens_fall', pct)} | "
            f"{cell(exp, 'core_false_per_h', num)} | {cell(exp, 'ofsyn_sens_fall', pct)} | "
            f"{cell(exp, 'elderly_sens_fall', pct)} | {cell(exp, 'occu_sens_fall', pct)} | "
            f"{cell(exp, 'best_false_per_h', num)} |"
        )
    lines += [
        "",
        "## Per run",
        "",
        (
            "| Run | t_high | Window AP | Macro F1 | Sens. FALL | False / h | Latency (s) "
            "| Core sens. | OF-Syn sens. | OCCU sens. | Best t | Sens. at best t | False / h at best t |"
        ),
        "| --- " * 13 + "|",
    ]
    for r in per_run.itertuples():
        lines.append(
            f"| {r.run} | {r.t_high} | {num(r.window_ap_falling)} | {num(r.window_macro_f1)} | "
            f"{pct(r.sens_fall)} | {num(r.false_per_h)} | {num(r.latency_s)} | "
            f"{pct(r.core_sens_fall)} | {pct(r.ofsyn_sens_fall)} | {pct(r.occu_sens_fall)} | "
            f"{num(r.best_t)} | {pct(r.best_sens)} | "
            f"{num(r.best_false_per_h)} |"
        )
    lines += [
        "",
        (
            f"Val holds {clips['duration_s'].sum() / 3600:.2f} h of video; false alerts per hour "
            "rest on very few normal-activity minutes, so read them as rough. The rule "
            "baseline has 0/1 scores, so its window AP and threshold sweep are not comparable."
        ),
    ]
    (EVAL / f"summary_{args.split}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"wrote {EVAL / f'summary_{args.split}.md'}")


if __name__ == "__main__":
    main()
