"""Evaluate one run on one split with the event metrics in fallsense.metrics.

    python scripts/08_evaluate.py --model e1_gru_s1 --split val
    python scripts/08_evaluate.py --model baseline --split val
    python scripts/08_evaluate.py --model e1_gru_s1 --split val --t-high 0.6 --t-low 0.3

Reads reports/scores/<model>/<split>.parquet (Contract 4), index.csv, OmniFall
labels and the run's thresholds.yaml (models/<model>/ or reports/scores/<model>/;
--t-high/--t-low override). Rows: the run with one threshold, the run with the
CHECK band, the rule baseline (if its scores exist for the split), and each of
those + tracker once reports/alerts/tracker/<split>.parquet (Contract 5) exists.
Writes reports/eval/eval_<model>_<split>.json, .md and _coverage.png (--out-dir).
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from fallsense import metrics, scores, windows
from fallsense.calibrate import CLASS_COLUMNS, alerts_from_scores
from fallsense.config import DATA, load_config, scores_path, windows_path
from fallsense.labels import load_labels

CFG = load_config()
REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports"
AGE = ["toddler", "child", "teen", "young adult", "middle-aged", "elderly"]
BMI = ["underweight", "normal", "overweight", "obese"]
HEADLINE = [
    ("sensitivity_alarm", "Sensitivity (FALL)", "pct"),
    ("sensitivity_any", "Sensitivity (FALL or CHECK)", "pct"),
    ("false_alarms_per_h", "False FALL alerts / h", "num"),
    ("checks_per_h", "CHECK requests / h", "num"),
    ("coverage", "Coverage", "pct"),
    ("median_latency_alarm_s", "Median latency (s)", "num"),
]


def find_thresholds(model: str, args: argparse.Namespace) -> dict[str, Any]:
    th: dict[str, Any] = {}
    for p in (REPO / "models" / model, scores_path(model, "x").parent):
        f = p / "thresholds.yaml"
        if f.exists():
            th = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            th["source"] = str(f.relative_to(REPO))
            break
    if args.t_high is not None:
        th.update(t_high=args.t_high, source="command line")
    if args.t_low is not None:
        th["t_low"] = args.t_low
    if "t_high" not in th:
        raise SystemExit(
            f"no thresholds.yaml for {model}; pass --t-high (and --t-low) explicitly"
        )
    th.setdefault("t_low", th["t_high"])
    return th


def evaluate(
    clips: pd.DataFrame, events: pd.DataFrame, alerts: pd.DataFrame, n_boot: int
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    clip_df, event_df = metrics.match_events(clips, events, alerts)
    res = metrics.summarise(clip_df, event_df)
    res["ci"] = metrics.bootstrap(clip_df, event_df, n_boot=n_boot)
    return res, clip_df, event_df


def clip_groups(clips: pd.DataFrame) -> dict[str, pd.Series]:
    """Breakdown columns: dataset, qc_pass, OF-Syn age and BMI, staged tags."""
    groups = {"dataset": clips["dataset"], "qc_pass": clips["qc_pass"].astype(str)}
    syn = clips["dataset"] == "of-syn"
    if syn.any():
        meta = load_labels("of-syn").drop_duplicates("path").set_index("path")
        paths = clips.loc[syn, "path"]
        groups["of-syn age"] = paths.map(meta["age_group"]).map(dict(enumerate(AGE)))
        groups["of-syn BMI"] = paths.map(meta["bmi_band"]).map(dict(enumerate(BMI)))
    staged = clips["dataset"] == "staged"
    if staged.any():
        tags = clips.loc[staged, "path"].str.lower().str.split("[_/]")
        groups["staged lighting"] = tags.map(
            lambda t: "dim" if "dim" in t else "normal"
        )
        groups["staged occluded"] = tags.map(lambda t: "yes" if "occ" in t else "no")
    return groups


def breakdown(
    clips: pd.DataFrame, clip_df: pd.DataFrame, event_df: pd.DataFrame, n_boot: int
) -> list[dict[str, Any]]:
    rows = []
    for name, col in clip_groups(clips).items():
        col = col.dropna()
        for value in sorted(col.unique(), key=str):
            ids = set(clips.loc[col.index[col == value], "clip_id"])
            c = clip_df[clip_df["clip_id"].isin(ids)]
            e = event_df[event_df["clip_id"].isin(ids)]
            res = metrics.summarise(c, e)
            res["ci"] = metrics.bootstrap(c, e, n_boot=n_boot)
            rows.append({"group": name, "value": str(value), **res})
    return rows


def coverage_curve(
    sc: pd.DataFrame, th: dict[str, Any], clips: pd.DataFrame, events: pd.DataFrame
) -> list[dict[str, float]]:
    """Sweep the CHECK band from 0 to t_high (t_low = t_high - width)."""
    out = []
    for width in np.arange(0, th["t_high"] + 1e-9, 0.05):
        t_low = max(0.0, th["t_high"] - width)
        alerts = alerts_from_scores(sc, {**th, "t_low": t_low})
        res = metrics.summarise(*metrics.match_events(clips, events, alerts))
        out.append(
            {
                "band": round(float(width), 2),
                "t_low": round(t_low, 2),
                "coverage": res["coverage"],
                "missed_falls": 1 - res["sensitivity_any"],
                "false_alarms_per_h": res["false_alarms_per_h"],
                "checks_per_h": res["checks_per_h"],
            }
        )
    return out


def plot_curve(curve: list[dict[str, float]], title: str, path: Path) -> None:
    df = pd.DataFrame(curve)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(df["coverage"], df["missed_falls"], "o-", label="missed falls (share)")
    ax.set_xlabel("coverage (decided without a CHECK)")
    ax.set_ylabel("missed falls")
    ax2 = ax.twinx()
    ax2.plot(df["coverage"], df["false_alarms_per_h"], "s--", color="tab:red")
    ax2.set_ylabel("false FALL alerts / h", color="tab:red")
    for r in df.itertuples():
        ax.annotate(f"{r.t_low:.2f}", (r.coverage, r.missed_falls), fontsize=7)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def window_report(sc: pd.DataFrame, split: str, path: Path | None) -> dict[str, Any]:
    p = path or windows_path(split)
    if not Path(p).exists():
        return {"note": f"no windows file at {p}"}
    w = windows.load_windows(split, path=Path(p))
    key = pd.DataFrame(
        {"clip_id": w["clip_id"], "end_frame": w["end_frame"], "y": w["y"]}
    )
    m = key.merge(sc, on=["clip_id", "end_frame"], how="inner")
    if m.empty:
        return {"note": "no score rows match the windows file"}
    pred = m[CLASS_COLUMNS].to_numpy().argmax(axis=1)
    return metrics.window_metrics(m["y"].to_numpy(), pred)


def fmt(res: dict[str, Any], key: str, kind: str) -> str:
    v = res.get(key, float("nan"))
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    lo_hi = res.get("ci", {}).get(key)

    def one(x: float) -> str:
        return f"{100 * x:.1f}%" if kind == "pct" else f"{x:.2f}"

    return one(v) + (f" ({one(lo_hi[0])}–{one(lo_hi[1])})" if lo_hi else "")


def write_markdown(report: dict[str, Any], path: Path) -> None:
    r = report
    lines = [
        f"# Evaluation: {r['model']} on {r['split']}",
        "",
        (
            f"Thresholds: t_high {r['thresholds']['t_high']}, t_low "
            f"{r['thresholds']['t_low']} ({r['thresholds'].get('source', '')}). "
            f"{r['clips']} clips, {r['events']} falls, {r['hours']:.2f} h "
            f"({r['normal_hours']:.2f} h normal activity); "
            f"{r['unscored_clips']} clips have no score rows (counted as silent). "
            "95% intervals: bootstrap over clips."
        ),
        "",
        "## Systems",
        "",
        "| System | " + " | ".join(h for _, h, _ in HEADLINE) + " | Alerts |",
        "| --- " * (len(HEADLINE) + 2) + "|",
    ]
    for name, res in r["systems"].items():
        cells = [fmt(res, k, kind) for k, _, kind in HEADLINE]
        lines.append(
            f"| {name} | " + " | ".join(cells) + f" | {res['alarms']} FALL, "
            f"{res['checks']} CHECK |"
        )
    lines += [
        "",
        f"## Breakdown ({r['breakdown_system']})",
        "",
        (
            "| Group | Value | Clips | Falls | Hours | Sens. (FALL) | Sens. (any) "
            "| False FALL / h | CHECK / h |"
        ),
        "| --- " * 9 + "|",
    ]
    for b in r["breakdown"]:
        lines.append(
            f"| {b['group']} | {b['value']} | {b['clips']} | {b['events']} | "
            f"{b['hours']:.2f} | {fmt(b, 'sensitivity_alarm', 'pct')} | "
            f"{fmt(b, 'sensitivity_any', 'pct')} | "
            f"{fmt(b, 'false_alarms_per_h', 'num')} | {fmt(b, 'checks_per_h', 'num')} |"
        )
    wm = r["windows"]
    lines += ["", "## Windows (4 classes)", ""]
    if "macro_f1" in wm:
        lines.append(f"Macro F1 {wm['macro_f1']:.3f} over {wm['windows']} windows.")
        lines += ["", "| true \\ predicted | " + " | ".join(CFG["classes"]) + " |"]
        lines.append("| --- " * 5 + "|")
        for name, row in zip(CFG["classes"], wm["confusion"]):
            lines.append(f"| {name} | " + " | ".join(map(str, row)) + " |")
    else:
        lines.append(wm.get("note", ""))
    if r.get("curve"):
        lines += [
            "",
            "## Coverage–error curve",
            "",
            f"![coverage]({Path(r['curve_png']).name})",
            "",
            "| CHECK band | t_low | Coverage | Missed falls | False FALL / h | CHECK / h |",
            "| --- " * 6 + "|",
        ]
        for c in r["curve"]:
            lines.append(
                f"| {c['band']:.2f} | {c['t_low']:.2f} | {100 * c['coverage']:.1f}% | "
                f"{100 * c['missed_falls']:.1f}% | {c['false_alarms_per_h']:.2f} | "
                f"{c['checks_per_h']:.2f} |"
            )
    lines += [
        "",
        (
            "Definitions: fall event = OmniFall `fall` segment; detected = alert from the "
            "segment start to 10 s after its end; false alert = alert matching no fall; "
            "normal-activity hours exclude [start − 2 s, end + 10 s]; FALL and ON FLOOR "
            "count as alarms, CHECK hands over to a person."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="score folder in reports/scores")
    ap.add_argument("--split", required=True, help="val, test, oops or staged_test")
    ap.add_argument("--index", type=Path, default=DATA / "skeletons" / "index.csv")
    ap.add_argument("--windows", type=Path, help="Contract 3 file for window F1")
    ap.add_argument("--t-high", type=float)
    ap.add_argument("--t-low", type=float)
    ap.add_argument("--boot", type=int, default=metrics.N_BOOT)
    ap.add_argument("--out-dir", type=Path, default=REPORTS / "eval")
    args = ap.parse_args()

    index = pd.read_csv(args.index)
    clips = metrics.eval_clips(index, args.split)
    events = metrics.fall_events(clips)
    ids = set(clips["clip_id"])
    sc = scores.load_scores(args.model, args.split)
    sc = sc[sc["clip_id"].isin(ids)]
    th = find_thresholds(args.model, args)
    one = {**th, "t_low": th["t_high"]}

    systems: dict[str, pd.DataFrame] = {
        f"{args.model} (one threshold)": alerts_from_scores(sc, one)
    }
    main_name = f"{args.model} (one threshold)"
    if th["t_low"] < th["t_high"]:
        main_name = f"{args.model} (CHECK band)"
        systems[main_name] = alerts_from_scores(sc, th)
    base_file = scores_path("baseline", args.split)
    if args.model != "baseline" and base_file.exists():
        base = scores.load_scores("baseline", args.split)
        base = base[base["clip_id"].isin(ids)]
        base_th = find_thresholds(
            "baseline", argparse.Namespace(t_high=None, t_low=None)
        )
        systems["rule baseline (E0)"] = alerts_from_scores(base, base_th)
    tracker = REPORTS / "alerts" / "tracker" / f"{args.split}.parquet"
    if tracker.exists():
        tr = pd.read_parquet(tracker)
        tr = tr[tr["clip_id"].isin(ids)]
        for name in list(systems):
            systems[f"{name} + tracker"] = pd.concat(
                [systems[name], tr[["clip_id", "t_s", "level"]]], ignore_index=True
            )

    report: dict[str, Any] = {
        "model": args.model,
        "split": args.split,
        "thresholds": {k: th[k] for k in ("t_high", "t_low", "source") if k in th},
        "unscored_clips": len(ids - set(sc["clip_id"])),
        "systems": {},
    }
    for name, alerts in systems.items():
        res, clip_df, event_df = evaluate(clips, events, alerts, args.boot)
        report["systems"][name] = res
        if name == main_name:
            report["breakdown_system"] = name
            report["breakdown"] = breakdown(clips, clip_df, event_df, args.boot)
            report.update(
                {k: res[k] for k in ("clips", "events", "hours", "normal_hours")}
            )
    report["windows"] = window_report(sc, args.split, args.windows)

    stem = f"eval_{args.model}_{args.split}"
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if th["t_high"] > 0:
        report["curve"] = coverage_curve(sc, th, clips, events)
        report["curve_png"] = f"{stem}_coverage.png"  # next to the .md
        plot_curve(
            report["curve"], f"{args.model} on {args.split}", out / report["curve_png"]
        )
    (out / f"{stem}.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )
    write_markdown(report, out / f"{stem}.md")
    for name, res in report["systems"].items():
        print(
            f"{name:34s} sens FALL {fmt(res, 'sensitivity_alarm', 'pct'):22s} "
            f"any {fmt(res, 'sensitivity_any', 'pct'):22s} "
            f"false/h {fmt(res, 'false_alarms_per_h', 'num')}"
        )
    print(f"wrote {out / stem}.md / .json")


if __name__ == "__main__":
    main()
