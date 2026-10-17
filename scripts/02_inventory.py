"""Build data/inventory.csv: one row per (dataset, path) with split, probe results and flags.

    python scripts/02_inventory.py --datasets le2i occu

Rows for datasets not named on the command line are kept from the previous run,
so the inventory can be built up one batch of datasets at a time.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from fallsense.config import DATA, load_config
from fallsense.labels import load_labels

CFG = load_config()
KNOWN = CFG["datasets"] + CFG["test_only_datasets"]
SINGLE_ACTOR = {"mcfd"}
FALL = 1
MIN_SECONDS = 2.0
FPS_RANGE = (10, 60)
FALLBACK_FPS = 25.0
OVERRUN_S = 0.5


def video_path(root: Path, dataset: str, path: str) -> Path:
    return root / dataset / "video" / f"{path}.mp4"


def probe(file: Path) -> dict[str, object]:
    if not file.exists():
        return {"flag": "missing", "reason": "file not found"}
    cap = cv2.VideoCapture(str(file))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out: dict[str, object] = {
            "fps": fps,
            "n_frames": n,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        ok_first, _ = cap.read()
        ok_last = False
        if ok_first and n > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
            ok_last, _ = cap.read()
    finally:
        cap.release()
    if not ok_first:
        return out | {"flag": "unreadable", "reason": "first frame did not decode"}
    if not ok_last:
        return out | {"flag": "unreadable", "reason": "last frame did not decode"}
    return out


def clips_for(dataset: str) -> pd.DataFrame:
    segs = load_labels(dataset)
    segs = segs.assign(is_fall=segs["label"] == FALL)
    clips = segs.groupby("path", sort=True).agg(
        split=("split", "first"),
        n_splits=("split", "nunique"),
        subject=("subject", "first"),
        cam=("cam", "first"),
        label_end=("end", "max"),
        n_segments=("label", "size"),
        n_fall_segments=("is_fall", "sum"),
    )
    clips = clips.reset_index()
    clips.insert(0, "dataset", dataset)
    return clips


def inventory(dataset: str, root: Path, workers: int) -> pd.DataFrame:
    clips = clips_for(dataset)
    files = [video_path(root, dataset, p) for p in clips["path"]]
    with ThreadPoolExecutor(workers) as pool:
        probes = list(
            tqdm(pool.map(probe, files), total=len(files), desc=dataset, unit="clip")
        )
    df = pd.concat([clips, pd.DataFrame(probes)], axis=1)
    if "flag" not in df:
        df["flag"] = None
        df["reason"] = None
    df["file"] = [str(f) for f in files]

    ok = df["flag"].isna()
    fps = df["fps"].fillna(0)
    odd = ok & ((fps < FPS_RANGE[0]) | (fps > FPS_RANGE[1]))
    df.loc[odd, ["flag", "reason"]] = [
        "fps_odd",
        f"fps outside {FPS_RANGE}; using {FALLBACK_FPS}",
    ]
    df["fps_used"] = fps.where(~odd & (fps > 0), FALLBACK_FPS)
    df["duration_s"] = df["n_frames"] / df["fps_used"]

    ok = df["flag"].isna()
    short = ok & (df["duration_s"] < MIN_SECONDS)
    df.loc[short, ["flag", "reason"]] = ["short", f"under {MIN_SECONDS} s"]

    ok = df["flag"].isna()
    overrun = ok & (df["label_end"] > df["duration_s"] + OVERRUN_S)
    df.loc[overrun, "flag"] = "labels_overrun"
    df.loc[overrun, "reason"] = (
        "labels end "
        + df["label_end"].round(2).astype(str)
        + " s > video "
        + df["duration_s"].round(2).astype(str)
        + " s"
    )

    multi = df["n_splits"] > 1
    if multi.any():
        print(
            f"WARNING {dataset}: {multi.sum()} clips appear in more than one split; first split kept"
        )
    return df.drop(columns="n_splits")


def summarise(df: pd.DataFrame) -> None:
    s = (
        df.groupby(["dataset", "split"])
        .agg(
            clips=("path", "size"),
            hours=("duration_s", lambda x: x.sum() / 3600),
            subjects=("subject", lambda x: x[x >= 0].nunique()),
            fall_segments=("n_fall_segments", "sum"),
            flagged=("flag", lambda x: x.notna().sum()),
        )
        .round({"hours": 2})
    )
    print(s.to_string())
    flags = df[df["flag"].notna()].groupby(["dataset", "flag"]).size()
    if len(flags):
        print("\nFlags:")
        print(flags.to_string())


def leak_check(df: pd.DataFrame) -> bool:
    ok = True
    known = df[df["subject"] >= 0]
    for key, g in known.groupby("dataset"):
        dataset = str(key)
        train = set(g.loc[g["split"] == "train", "subject"])
        test = set(g.loc[g["split"] == "test", "subject"])
        shared = sorted(train & test)
        if not shared:
            continue
        if dataset in SINGLE_ACTOR:
            print(
                f"WARNING {dataset}: single actor, subjects {shared} in train and test; "
                "OmniFall split accepted (note in DATA_CARD.md)"
            )
        else:
            print(f"LEAK {dataset}: subjects {shared} are in both train and test")
            ok = False
    return ok


def main() -> int:
    root = Path(os.environ["OMNIFALL_ROOT"])
    out = DATA / "inventory.csv"
    on_disk = [d for d in KNOWN if (root / d / "video").is_dir()]

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=on_disk, choices=KNOWN)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    new = pd.concat(
        [inventory(d, root, args.workers) for d in args.datasets], ignore_index=True
    )
    if out.exists():
        old = pd.read_csv(out)
        new = pd.concat(
            [old[~old["dataset"].isin(args.datasets)], new], ignore_index=True
        )
    new = new.sort_values(["dataset", "path"]).reset_index(drop=True)
    new.to_csv(out, index=False)
    print(f"\nWrote {out} ({len(new)} rows)\n")

    summarise(new)
    if not leak_check(new):
        print("\nFAIL: subject leak between train and test")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
