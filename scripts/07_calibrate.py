"""Calibrate one run on val: temperature, then FALL and CHECK thresholds.

    python scripts/07_calibrate.py --run e3_tcn_s1
    python scripts/07_calibrate.py --run e3_tcn_s1 --target-fall 0.85 --target-any 0.95

1. Temperature: one scalar T fitted on models/<run>/val_logits.npy by minimising
   NLL (LBFGS from T = 1); expected calibration error (15 bins) before and after.
2. Rewrites reports/scores/<run>/val.parquet with T (calibrate.write_scores).
3. Thresholds on val, sweeping 0.05-0.95 in 0.05 steps with alerts_from_scores
   and Dev A's metrics:
   - t_high = the highest threshold whose FALL sensitivity is >= --target-fall
     (val holds about 45 min of normal activity, too little for the plan's
     "<= 0.5 false FALLs per hour" rule, so we fix sensitivity and report the
     false-alert rate with its bootstrap interval instead);
   - t_low = the highest threshold <= t_high whose FALL-or-CHECK sensitivity is
     >= --target-any; if none reaches it, the one with the best sensitivity.
4. Writes models/<run>/thresholds.yaml (read by 08_evaluate.py, the live demo and
   write_scores) and models/<run>/calibration.csv (the sweep).

Only val is used. Test, staged_test and OOPS stay closed until the model is frozen.
"""

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fallsense import calibrate, metrics, windows
from fallsense.config import DATA

GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
ECE_BINS = 15


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Scalar T minimising NLL of softmax(logits / T)."""
    x = torch.from_numpy(logits.astype(np.float64))
    t = torch.from_numpy(y.astype(np.int64))
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)  # T = exp(log_t)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)
    nll = torch.nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = nll(x / log_t.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def ece(probs: np.ndarray, y: np.ndarray, bins: int = ECE_BINS) -> float:
    """Expected calibration error of the top-class confidence."""
    conf = probs.max(axis=1)
    right = probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in itertools.pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(right[m].mean() - conf[m].mean())
    return float(total)


def event_metrics(
    sc: pd.DataFrame,
    clips: pd.DataFrame,
    events: pd.DataFrame,
    t_high: float,
    t_low: float,
    boot: int = 0,
) -> dict[str, Any]:
    alerts = calibrate.alerts_from_scores(sc, {"t_high": t_high, "t_low": t_low})
    clip_df, event_df = metrics.match_events(clips, events, alerts)
    res: dict[str, Any] = metrics.summarise(clip_df, event_df)
    if boot:
        res["ci"] = metrics.bootstrap(clip_df, event_df, n_boot=boot)
    return res


def sweep(sc: pd.DataFrame, clips: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """One threshold for FALL (no CHECK band) across the grid."""
    rows = []
    for t in GRID:
        r = event_metrics(sc, clips, events, t, t)
        rows.append(
            {
                "t": t,
                "sens_fall": r["sensitivity_alarm"],
                "false_fall_per_h": r["false_alarms_per_h"],
            }
        )
    return pd.DataFrame(rows)


def pick_t_high(
    table: pd.DataFrame, target: float, plateau: float = 0.03
) -> tuple[float, bool]:
    """Highest threshold meeting the FALL sensitivity target (fewest false alerts).

    If no threshold reaches it, sensitivity has plateaued: the last few points
    come from flooding clips with alerts, so take the highest threshold within
    `plateau` of the best achievable sensitivity instead.
    """
    ok = table[table["sens_fall"] >= target]
    if len(ok):
        return float(ok["t"].max()), True
    near = table[table["sens_fall"] >= table["sens_fall"].max() - plateau - 1e-9]
    return float(near["t"].max()), False


def pick_t_low(
    sc: pd.DataFrame,
    clips: pd.DataFrame,
    events: pd.DataFrame,
    t_high: float,
    target: float,
    plateau: float = 0.03,
) -> tuple[float, bool, pd.DataFrame]:
    """Highest t_low <= t_high whose FALL-or-CHECK sensitivity meets `target`
    (if unreachable: highest t_low within `plateau` of the best)."""
    rows = []
    for t in GRID[GRID <= t_high][::-1]:
        r = event_metrics(sc, clips, events, t_high, t)
        rows.append(
            {
                "t_low": t,
                "sens_any": r["sensitivity_any"],
                "checks_per_h": r["checks_per_h"],
            }
        )
        if r["sensitivity_any"] >= target:
            return float(t), True, pd.DataFrame(rows)
    table = pd.DataFrame(rows)
    if not len(table):
        return t_high, False, table
    top = table[table["sens_any"] >= table["sens_any"].max() - plateau - 1e-9]
    best = top["t_low"].max()
    return float(best), False, table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="folder name in models/")
    ap.add_argument("--target-fall", type=float, default=0.90)
    ap.add_argument("--target-any", type=float, default=0.95)
    ap.add_argument(
        "--plateau",
        type=float,
        default=0.03,
        help="if --target-fall is unreachable: stay within this of the best sensitivity",
    )
    ap.add_argument("--windows-dir", type=Path, default=DATA / "windows")
    ap.add_argument("--index", type=Path, default=DATA / "skeletons" / "index.csv")
    ap.add_argument("--boot", type=int, default=metrics.N_BOOT)
    args = ap.parse_args()

    run_dir = REPO / "models" / args.run
    val_npz = args.windows_dir / "val.npz"
    y = windows.load_windows("val", path=val_npz)["y"]
    logits = np.load(run_dir / "val_logits.npy")
    if len(logits) != len(y):
        raise SystemExit(
            f"val_logits.npy has {len(logits)} rows but val.npz has {len(y)}; "
            "retrain the run on the current windows"
        )

    temperature = fit_temperature(logits, y)
    ece_before = ece(calibrate.softmax_t(logits, 1.0), y)
    ece_after = ece(calibrate.softmax_t(logits, temperature), y)
    print(f"temperature {temperature:.3f}; ECE {ece_before:.4f} -> {ece_after:.4f}")

    sc = calibrate.write_scores(run_dir, val_npz, temperature=temperature)
    index = pd.read_csv(args.index)
    clips = metrics.eval_clips(index, "val")
    events = metrics.fall_events(clips)
    sc = sc[sc["clip_id"].isin(set(clips["clip_id"]))]

    table = sweep(sc, clips, events)
    t_high, high_ok = pick_t_high(table, args.target_fall, args.plateau)
    t_low, low_ok, low_table = pick_t_low(
        sc, clips, events, t_high, args.target_any, args.plateau
    )
    final = event_metrics(sc, clips, events, t_high, t_low, boot=args.boot)
    ci = final["ci"]

    def with_ci(key: str, pct: bool = False) -> str:
        k = 100 if pct else 1
        lo, hi = ci.get(key, (np.nan, np.nan))
        unit = "%" if pct else ""
        return f"{final[key] * k:.1f}{unit} ({lo * k:.1f}-{hi * k:.1f})"

    notes = []
    if not high_ok:
        notes.append(
            f"no threshold reaches FALL sensitivity {args.target_fall:.0%}; "
            f"highest threshold within {args.plateau:.0%} of the best "
            f"({table['sens_fall'].max():.1%})"
        )
    if not low_ok:
        notes.append(
            f"no t_low reaches FALL-or-CHECK sensitivity {args.target_any:.0%}"
        )
    thresholds = {
        "temperature": round(temperature, 4),
        "t_high": t_high,
        "t_low": t_low,
        "on_floor_seconds_eval": 5,
        "on_floor_seconds_demo": 20,
        "calibration": {
            "split": "val",
            "target_fall": args.target_fall,
            "target_any": args.target_any,
            "plateau": args.plateau,
            "ece_before": round(ece_before, 4),
            "ece_after": round(ece_after, 4),
            "sens_fall": round(final["sensitivity_alarm"], 4),
            "sens_any": round(final["sensitivity_any"], 4),
            "false_fall_per_h": round(final["false_alarms_per_h"], 2),
            "false_fall_per_h_ci": [round(v, 2) for v in ci["false_alarms_per_h"]],
            "checks_per_h": round(final["checks_per_h"], 2),
            "normal_hours": round(final["normal_hours"], 3),
            "notes": notes,
        },
    }
    with open(run_dir / "thresholds.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(thresholds, f, sort_keys=False)
    table.merge(low_table.rename(columns={"t_low": "t"}), on="t", how="left").to_csv(
        run_dir / "calibration.csv", index=False
    )

    print(
        f"t_high {t_high:.2f}, t_low {t_low:.2f}"
        + (f"  ({'; '.join(notes)})" if notes else "")
    )
    print(
        f"val: FALL sensitivity {with_ci('sensitivity_alarm', True)}, "
        f"FALL or CHECK {with_ci('sensitivity_any', True)}, "
        f"false FALL/h {with_ci('false_alarms_per_h')}, "
        f"CHECK/h {final['checks_per_h']:.1f}"
    )
    print(f"wrote {run_dir / 'thresholds.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
