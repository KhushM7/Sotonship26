"""Collect models/e<row>_<model>_s<seed>/ runs into reports/experiments.csv.

    python scripts/experiments_table.py

One row per experiment and model (e.g. e4_gru, e4_tcn): seeds, run names, the
git version and laptop each run was trained on, how many runs are calibrated
(models/<run>/thresholds.yaml), and mean/sd of the best epoch's val metrics from
train_log.csv (val_* on the train-set val windows used for selection, val_all_*
on all of val). Event metrics live in reports/eval/summary_val.md (Dev A).
"""

import re
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = []
    for run in sorted(REPO.glob("models/e*_*_s*")):
        m = re.fullmatch(r"(e\d+)_(\w+)_s(\d+)", run.name)
        log = run / "train_log.csv"
        if not m or not log.exists() or not (run / "val_logits.npy").exists():
            continue  # unfinished runs are left out
        cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))["run"]
        df = pd.read_csv(log)
        best = df.sort_values(["val_ap_falling", "val_macro_f1"]).iloc[-1]
        rows.append(
            {
                "exp": f"{m[1]}_{cfg['model']}",
                "run": run.name,
                "model": cfg["model"],
                "train_sets": " ".join(cfg["train_sets"] or ["all"]),
                "augment": cfg["augment"],
                "occlude": cfg.get("occlude", False),
                "seed": cfg["seed"],
                "git_commit": cfg.get("git_commit", ""),
                "laptop": cfg.get("laptop", ""),
                "val_selection": cfg.get("val_selection", ""),
                "best_epoch": int(best["epoch"]),
                "val_ap_falling": best["val_ap_falling"],
                "val_macro_f1": best["val_macro_f1"],
                "val_all_ap_falling": best.get("val_all_ap_falling"),
                "calibrated": (run / "thresholds.yaml").exists(),
                "s_per_epoch": df["seconds"].iloc[1:].median(),
            }
        )
    runs = pd.DataFrame(rows)
    g = runs.groupby("exp")
    cols = ["model", "train_sets", "augment", "occlude"]
    out = g[[*cols, "git_commit", "laptop", "val_selection"]].first()
    out["seeds"] = g["seed"].count()
    out["runs"] = g["run"].agg(" ".join)
    out["calibrated"] = g["calibrated"].sum()
    for col in ["val_ap_falling", "val_macro_f1", "val_all_ap_falling"]:
        out[f"{col}_mean"] = g[col].mean().round(4)
        out[f"{col}_sd"] = g[col].std().round(4)
    out["best_epoch_mean"] = g["best_epoch"].mean().round(1)
    out["s_per_epoch"] = g["s_per_epoch"].median().round(1)
    out.reset_index().to_csv(REPO / "reports" / "experiments.csv", index=False)
    print(out.drop(columns=["runs"]).to_string())


if __name__ == "__main__":
    main()
