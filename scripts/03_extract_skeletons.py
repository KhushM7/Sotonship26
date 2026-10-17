"""Extract MediaPipe skeletons for every usable clip in data/inventory.csv (Contract 1).

    python scripts/03_extract_skeletons.py --datasets le2i --limit 5

Resumable: clips whose .npz already exists are skipped unless --overwrite.
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import mediapipe as mp
import numpy as np
import pandas as pd

from fallsense.config import DATA, load_config, skeleton_path
from fallsense.io_video import iter_frames
from fallsense.pose import F32, PoseExtractor

CFG = load_config()
FPS = CFG["fps"]
POSE_MODEL = Path(__file__).resolve().parents[1] / CFG["pose_model"]
POSE = {k: CFG["pose"][k] for k in ("min_det", "min_presence", "min_track")}
LE2I_UPSCALE = CFG["le2i_upscale"]

SKIP_FLAGS = {"missing", "unreadable"}
LOG_FIELDS = ["dataset", "path", "frames", "det_rate", "seconds", "pose_model"]


def extract_clip(
    row: dict[str, Any], out: str, model: str, upscale: float
) -> dict[str, Any]:
    t0 = time.perf_counter()
    kps: list[F32] = []
    worlds: list[F32] = []
    dets: list[bool] = []
    ts: list[int] = []
    with PoseExtractor(model, upscale=upscale, **POSE) as pose:
        for t_ms, frame in iter_frames(row["file"], FPS):
            kp, w, det = pose(frame, t_ms)
            kps.append(kp)
            worlds.append(w)
            dets.append(det)
            ts.append(t_ms)
    if not ts:
        raise RuntimeError(f"no frames decoded from {row['file']}")
    detected = np.array(dets, bool)
    pose_model = f"{Path(model).stem}@{mp.__version__}" + (
        f"x{upscale:g}" if upscale != 1 else ""
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            kp=np.stack(kps).astype(np.float32),
            world=np.stack(worlds).astype(np.float32),
            detected=detected,
            t_ms=np.array(ts, np.int64),
            fps=np.int64(FPS),
            width=np.int64(row["width"]),
            height=np.int64(row["height"]),
            dataset=np.str_(row["dataset"]),
            path=np.str_(row["path"]),
            subject=np.int64(row["subject"]),
            cam=np.int64(row["cam"]),
            pose_model=np.str_(pose_model),
        )
    os.replace(tmp, out_path)
    return {
        "dataset": row["dataset"],
        "path": row["path"],
        "frames": len(ts),
        "det_rate": round(float(detected.mean()), 4),
        "seconds": round(time.perf_counter() - t0, 2),
        "pose_model": pose_model,
    }


def main() -> int:
    data = DATA
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, help="at most N clips per dataset")
    ap.add_argument(
        "--clip-list", type=Path, help="CSV with a `path` column; only these clips"
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--pose-model", type=Path, default=POSE_MODEL)
    ap.add_argument("--log", type=Path, default=Path("reports/extract_log.csv"))
    args = ap.parse_args()

    inv = pd.read_csv(data / "inventory.csv")
    inv = inv[~inv["flag"].isin(SKIP_FLAGS)]
    if args.datasets:
        inv = inv[inv["dataset"].isin(args.datasets)]
    if args.clip_list:
        inv = inv[inv["path"].isin(set(pd.read_csv(args.clip_list)["path"]))]
    if args.limit:
        inv = inv.groupby("dataset", group_keys=False).head(args.limit)
    inv = inv.assign(
        out=[str(skeleton_path(d, p)) for d, p in zip(inv["dataset"], inv["path"])]
    )
    if not args.overwrite:
        inv = inv[[not Path(o).exists() for o in inv["out"]]]
    if inv.empty:
        print("Nothing to do: every selected clip already has a skeleton file.")
        return 0

    total_s = inv["duration_s"].sum()
    print(
        f"{len(inv)} clips, {total_s / 3600:.2f} h of video, {args.workers} workers, "
        f"model {args.pose_model.name}"
    )

    args.log.parent.mkdir(parents=True, exist_ok=True)
    new_log = not args.log.exists()
    failures: list[tuple[str, str, str]] = []
    t0 = time.perf_counter()
    done_s = 0.0
    with (
        open(args.log, "a", newline="") as logf,
        ProcessPoolExecutor(args.workers) as pool,
    ):
        log = csv.DictWriter(logf, LOG_FIELDS)
        if new_log:
            log.writeheader()
        futures: dict[Future[dict[str, Any]], dict[str, Any]] = {}
        rows = cast("list[dict[str, Any]]", inv.to_dict("records"))  # str columns
        for row in rows:
            upscale = 2.0 if (row["dataset"] == "le2i" and LE2I_UPSCALE) else 1.0
            fut = pool.submit(
                extract_clip, row, row["out"], str(args.pose_model), upscale
            )
            futures[fut] = row
        for i, fut in enumerate(as_completed(futures), 1):
            row = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001 - keep going; report at the end
                failures.append((row["dataset"], row["path"], repr(e)))
                continue
            log.writerow(rec)
            logf.flush()
            done_s += row["duration_s"]
            if i == 10 or i % 50 == 0 or i == len(futures):
                elapsed = time.perf_counter() - t0
                eta = elapsed / done_s * (total_s - done_s) if done_s else float("nan")
                print(
                    f"[{i}/{len(futures)}] {elapsed / 60:.1f} min elapsed, ETA {eta / 60:.1f} min "
                    f"(last: {rec['path']} det {rec['det_rate']:.2f})",
                    flush=True,
                )

    for d, p, err in failures:
        print(f"FAILED {d}/{p}: {err}")
    print(f"Done: {len(futures) - len(failures)} ok, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
