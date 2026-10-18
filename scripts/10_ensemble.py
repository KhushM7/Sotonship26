"""Make an ensemble run that averages the probabilities of trained member runs.

    python scripts/10_ensemble.py --name e9_ens_gru --members e9_gru_s1 e9_gru_s2 e9_gru_s3

Writes models/<name>/: ensemble.yaml (members), val_logits.npy (log of the mean
member softmax, so 07_calibrate.py fits one temperature and the thresholds on the
ensemble) and config.yaml (members, git commit, laptop). After that the folder
works like any run: 07_calibrate.py, 08_evaluate.py, calibrate.write_scores and
09_live.py all go through calibrate.Scorer, which reads ensemble.yaml.
"""

import argparse
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fallsense.calibrate import ensemble_logits


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True, help="new folder name in models/")
    ap.add_argument("--members", nargs="+", required=True, help="runs in models/")
    args = ap.parse_args()

    models = REPO / "models"
    logits, commits = [], []
    for m in args.members:
        d = models / m
        for f in ("model.pt", "feat_stats.npz", "val_logits.npy", "config.yaml"):
            if not (d / f).exists():
                raise SystemExit(f"member {m} has no {f}")
        logits.append(np.load(d / "val_logits.npy"))
        run = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))["run"]
        commits.append(str(run.get("git_commit", "")))
    if len({len(lg) for lg in logits}) != 1:
        raise SystemExit("members have different val_logits lengths")

    out = models / args.name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ensemble.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"members": args.members}, f, sort_keys=False)
    np.save(out / "val_logits.npy", ensemble_logits(logits))
    first = yaml.safe_load(
        (models / args.members[0] / "config.yaml").read_text("utf-8")
    )
    run = {
        "ensemble": args.members,
        "member_commits": commits,
        "git_commit": git_commit(),
        "laptop": platform.node(),
    }
    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"run": run, "config": first["config"]}, f, sort_keys=False)
    print(f"ensemble of {len(args.members)} runs -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
