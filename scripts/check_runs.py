"""Completion check for trained runs (handoff section 4.1).

    python scripts/check_runs.py e5_gru_s1 e5_tcn_s1 ...

Prints one PASS/FAIL line per item for each run, then the evidence lines from
logs/<run>.{train,calib,eval}.log when they exist. Exits 1 if any item fails.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fallsense.scores import validate_scores

REPO = Path(__file__).resolve().parents[1]
N_VAL = 12454
ACCEPTED_DIRTY = {"23288a1-dirty"}  # Dev B's E1-E4 runs
EVIDENCE = {
    "train": r"best val AP\(falling\)",
    "calib": r"^temperature .*ECE|val: FALL sensitivity",
    "eval": r"\((CHECK band|one threshold)\)",
}


def mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else -1.0


def check_run(run: str) -> list[tuple[bool, str]]:
    m = REPO / "models" / run
    out: list[tuple[bool, str]] = []

    def item(ok: bool, what: str) -> None:
        out.append((bool(ok), what))

    for name in ("model.pt", "feat_stats.npz", "confusion.png", "calibration.csv"):
        item((m / name).exists(), f"models/{run}/{name} exists")

    cfg_p = m / "config.yaml"
    commit = ""
    try:
        run_cfg = yaml.safe_load(cfg_p.read_text())["run"]
        commit = str(run_cfg.get("git_commit", ""))
        clean = not commit.endswith("-dirty") or commit in ACCEPTED_DIRTY
        item(clean, f"config.yaml git_commit {commit!r} clean")
        item(
            bool(run_cfg.get("laptop")), f"config.yaml laptop {run_cfg.get('laptop')!r}"
        )
    except Exception as e:  # noqa: BLE001
        item(False, f"config.yaml readable ({e})")

    try:
        cols = pd.read_csv(m / "train_log.csv", nrows=1).columns
        item(
            any(c.startswith("val_all_") for c in cols),
            "train_log.csv has val_all_* columns",
        )
    except Exception as e:  # noqa: BLE001
        item(False, f"train_log.csv readable ({e})")

    logits_p = m / "val_logits.npy"
    try:
        n = len(np.load(logits_p, mmap_mode="r"))
        item(n == N_VAL, f"val_logits.npy rows {n} == {N_VAL}")
    except Exception as e:  # noqa: BLE001
        item(False, f"val_logits.npy readable ({e})")

    thr_p = m / "thresholds.yaml"
    try:
        thr = yaml.safe_load(thr_p.read_text())
        item(
            isinstance(thr.get("calibration"), dict),
            "thresholds.yaml has calibration block",
        )
    except Exception as e:  # noqa: BLE001
        item(False, f"thresholds.yaml readable ({e})")

    sc_p = REPO / "reports" / "scores" / run / "val.parquet"
    try:
        sc = pd.read_parquet(sc_p)
        item(len(sc) == N_VAL, f"val.parquet rows {len(sc)} == {N_VAL}")
        try:
            validate_scores(sc)
            item(True, "val.parquet passes validate_scores")
        except ValueError as e:
            item(False, f"val.parquet validate_scores: {e}")
        item(mtime(sc_p) > mtime(logits_p), "val.parquet newer than val_logits.npy")
    except Exception as e:  # noqa: BLE001
        item(False, f"val.parquet readable ({e})")

    ev = REPO / "reports" / "eval"
    # Dev B's unzipped runs have a thresholds.yaml newer than their tracked eval
    # reports (unzip time vs checkout time); for those, compare by content instead
    same = commit in ACCEPTED_DIRTY and same_thresholds(
        ev / f"eval_{run}_val.json", thr_p
    )
    for suffix in ("_val.md", "_val.json", "_val_coverage.png"):
        p = ev / f"eval_{run}{suffix}"
        if mtime(p) > mtime(thr_p) > 0:
            item(True, f"eval_{run}{suffix} newer than thresholds.yaml")
        else:
            item(
                same and p.exists(),
                f"eval_{run}{suffix} newer than thresholds.yaml"
                + (" (Dev B run: same thresholds by content)" if same else ""),
            )
    return out


def same_thresholds(eval_json: Path, thr_p: Path) -> bool:
    try:
        used = json.loads(eval_json.read_text())["thresholds"]
        thr = yaml.safe_load(thr_p.read_text())
        return all(abs(used[k] - thr[k]) < 1e-9 for k in ("t_high", "t_low"))
    except Exception:  # noqa: BLE001
        return False


def evidence(run: str) -> list[str]:
    lines: list[str] = []
    for step, pat in EVIDENCE.items():
        log = REPO / "logs" / f"{run}.{step}.log"
        if not log.exists():
            continue
        for line in log.read_text(errors="replace").splitlines():
            if re.search(pat, line):
                lines.append(f"  [{step}] {line.strip()}")
    return lines


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    bad = 0
    for run in argv:
        results = check_run(run)
        n_fail = sum(not ok for ok, _ in results)
        bad += n_fail
        print(f"== {run}: {'PASS' if n_fail == 0 else f'FAIL ({n_fail})'}")
        for ok, what in results:
            print(f"  {'PASS' if ok else 'FAIL'}  {what}")
        print("\n".join(evidence(run)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
