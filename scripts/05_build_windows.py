"""Build Contract 3 v2 window arrays from skeletons, labels and the QC'd index.

    python scripts/05_build_windows.py                    # all splits from index.csv
    python scripts/05_build_windows.py --from-inventory --datasets GMDCSA24
                                                          # dry run before S7

Train and val use qc_pass clips only; test and OOPS use every clip
(clip_id joins back to index.csv, which keeps qc_pass). Only train uses the
dense near-fall stride; val is cut every stride_eval frames like test. OOPS goes to its own
oops.npz so it is never mixed into test. The dry run builds a provisional index
from data/inventory.csv (qc_pass = True) and writes to data/windows_dryrun/.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fallsense import labels, windows
from fallsense.config import DATA, load_config, skeleton_path

CFG = load_config()
CAP_S = CFG["occlusion"]["ctx_cap_s"]
TEST_ONLY = set(CFG["test_only_datasets"])
TRAIN_SPLITS = {"train", "val"}
INDEX_COLS = ["dataset", "path", "subject", "cam", "split", "duration_s", "qc_pass"]


def provisional_index(datasets: list[str] | None) -> pd.DataFrame:
    """index.csv stand-in from inventory.csv, for dry runs before S7."""
    inv = pd.read_csv(DATA / "inventory.csv")
    inv = inv[~inv["flag"].isin(["missing", "unreadable"])]
    if datasets:
        inv = inv[inv["dataset"].isin(datasets)]
    inv = inv[
        [skeleton_path(d, p).exists() for d, p in zip(inv["dataset"], inv["path"])]
    ]
    return inv.assign(qc_pass=True)[INDEX_COLS].reset_index(drop=True)


def output_split(dataset: str, split: str) -> str:
    return "oops" if dataset in TEST_ONLY else split


def build_clip(row: pd.Series, clip_id: int, counts: Counter) -> dict[str, np.ndarray]:
    d = np.load(skeleton_path(row["dataset"], row["path"]))
    width, height = int(d["width"]), int(d["height"])
    kp = windows.select_keypoints(d["kp"], d["detected"], width, height)
    present = d["detected"].astype(bool)
    wh = np.array([width, height])
    # context over the whole clip, then sliced: a window alone cannot know
    # someone vanished 8 s ago
    ctx = windows.last_seen_context(kp, present, wh, CFG["fps"], CAP_S)

    segs = labels.segments_for(row["dataset"], row["path"])
    segs = labels.clean_segments(segs, float(row["duration_s"]), counts)
    y4 = labels.to4(labels.frame_labels(segs, d["t_ms"]))

    # only train gets the dense near-fall stride; val is cut like test (every
    # stride_eval frames) so event metrics and calibration are not flattered
    train = row["split"] == "train"
    ends = windows.window_ends(len(y4), y4, train=train)
    out = windows.clip_windows(kp, present, ctx, y4, ends, CAP_S)
    n = len(out["y"])
    counts["windows_dropped"] += len(ends) - n
    out["wh"] = np.tile(wh, (n, 1))
    out["clip_id"] = np.full(n, clip_id)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--index", type=Path, default=DATA / "skeletons" / "index.csv")
    ap.add_argument("--datasets", nargs="+", help="only these datasets")
    ap.add_argument("--from-inventory", action="store_true", help="dry run before S7")
    ap.add_argument("--out", type=Path, help="output folder (default data/windows)")
    ap.add_argument(
        "--summary", type=Path, default=REPO / "reports" / "windows_summary.csv"
    )
    args = ap.parse_args()

    if args.from_inventory:
        index = provisional_index(args.datasets)
        out_dir = args.out or DATA / "windows_dryrun"
        print(f"DRY RUN: provisional index from inventory.csv ({len(index)} clips)")
    else:
        index = pd.read_csv(args.index)
        out_dir = args.out
    windows.check_splits(index)
    if args.datasets and not args.from_inventory:
        index = index[index["dataset"].isin(args.datasets)]  # keep clip_id = row number

    use = index[~index["split"].isin(TRAIN_SPLITS) | index["qc_pass"].astype(bool)]
    print(f"{len(use)} of {len(index)} clips used (train/val need qc_pass)")

    parts: dict[str, list[dict[str, np.ndarray]]] = {}
    counts: Counter = Counter()
    for clip_id, row in tqdm(list(use.iterrows()), unit="clip"):
        out = build_clip(row, int(clip_id), counts)
        out["dataset"] = np.full(len(out["y"]), row["dataset"])
        parts.setdefault(output_split(row["dataset"], row["split"]), []).append(out)

    summary = []
    for split, chunks in sorted(parts.items()):
        arrays = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        ds = arrays.pop("dataset")
        path = None if out_dir is None else out_dir / f"{split}.npz"
        windows.save_windows(split, **arrays, path=path)
        for (dset, y), n in Counter(zip(ds, arrays["y"])).items():
            summary.append(
                {"split": split, "dataset": dset, "class": CFG["classes"][y], "n": n}
            )
        print(f"{split}: {len(arrays['y'])} windows from {len(chunks)} clips")

    # every split's clip_ids are disjoint from the others
    seen: dict[int, str] = {}
    for split, chunks in parts.items():
        for c in chunks:
            for cid in np.unique(c["clip_id"]):
                if seen.setdefault(int(cid), split) != split:
                    raise ValueError(f"clip_id {cid} in {seen[int(cid)]} and {split}")

    table = pd.DataFrame(summary)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.summary, index=False)
    if len(table):
        print(
            table.pivot_table(
                "n", ["split", "dataset"], "class", aggfunc="sum", fill_value=0
            )
        )
        train = table[table["split"] == "train"]
        if len(train):
            share = train.loc[train["class"] == "falling", "n"].sum() / train["n"].sum()
            print(f"falling share of train windows: {share:.1%}")
            if share < 0.01:
                print(
                    "WARNING: falling under 1% of train; lower stride_train_near_fall to 1"
                )
    print("label cleaning:", dict(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
