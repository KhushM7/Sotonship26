"""QC every skeleton and write the clip index (Contract 2 v2).

    python scripts/04_qc_skeletons.py                 # all datasets, plots, 10 samples
    python scripts/04_qc_skeletons.py --samples 0     # skip the sample videos
    python scripts/04_qc_skeletons.py --clip of-syn:fall/fall_mi_062   # one clip

Sample videos draw the skeleton (coloured by labelled class) over the source video
where it is on this laptop (data/inventory.csv), else on black; --black forces black.

Needs only the .npz files and the OmniFall label configs (no video), so it works on
skeletons from either laptop. Writes data/skeletons/index.csv, reports/qc/*.png,
reports/qc/excluded.csv and reports/qc/samples/*.mp4.

A frame is *covered* if the pose was found or the frame lies inside a `hidden_down`
gap (person dropped out of sight away from the borders). `det_rate_covered` also
counts frames where the person is off-screen (before walking in / after walking out
at a border), so an empty room is not a detection failure; `det_rate_fall_covered`
does not. Train and val clips need `qc_pass`; test clips are all kept
(05_build_windows.py applies that rule).
"""

import argparse
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from fallsense.config import DATA, load_config, skeleton_path
from fallsense.io_video import iter_frames
from fallsense.labels import clean_segments, frame_labels, load_labels, to4
from fallsense.occlusion import HIPS, OCC, find_disappearances, stable_presence

CFG = load_config()
FPS = CFG["fps"]
KEYPOINTS: list[int] = CFG["keypoints"]
CLASSES: list[str] = CFG["classes"]
DATASETS: list[str] = [*CFG["datasets"], *CFG["test_only_datasets"]]
REPO = Path(__file__).resolve().parents[1]
QC_DIR = REPO / "reports" / "qc"

MIN_COVERED = 0.6
MIN_COVERED_FALL = 0.5
ID_JUMP = 0.25  # hip jump between detected frames, share of the image diagonal
MAX_ID_JUMPS = 3
MAX_STEP = 2  # frames between the two detections compared
FALL = (1, 2)  # falling, fallen

INDEX_COLUMNS = [
    "dataset",
    "path",
    "subject",
    "cam",
    "split",
    "n_frames",
    "duration_s",
    "width",
    "height",
    "det_rate",
    "det_rate_fall",
    "det_rate_covered",
    "det_rate_fall_covered",
    "hidden_frac_fall",
    "n_hidden_gaps",
    "id_jumps",
    "qc_pass",
    "qc_reason",
]

# 33 MediaPipe landmarks -> lines drawn in the sample videos
BONES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32),
    (0, 11), (0, 12),
]  # fmt: skip
BAR = 30  # header/footer height in the sample videos, px
# label -> (text, BGR colour) for the sample videos
CLASS_STYLE = {
    0: ("upright", (0, 200, 0)),
    1: ("FALLING", (0, 0, 255)),
    2: ("FALLEN", (0, 140, 255)),
    3: ("controlled_low", (255, 200, 0)),
}
UNLABELLED_STYLE = ("unlabelled", (160, 160, 160))


def load_kp(z: Any) -> tuple[npt.NDArray[np.float32], tuple[int, int]]:
    """Config keypoints as (T, 13, 3) pixel x, y, visibility, NaN where not found."""
    wh = (int(z["width"]), int(z["height"]))
    kp = z["kp"][:, KEYPOINTS][:, :, [0, 1, 3]].astype(np.float32)
    kp[..., 0] *= wh[0]
    kp[..., 1] *= wh[1]
    kp[~z["detected"].astype(bool)] = np.nan
    return kp, wh


def id_jumps(kp: npt.NDArray[np.float32], wh: tuple[int, int]) -> int:
    """Hip-midpoint jumps over 25% of the diagonal between detections at most
    MAX_STEP frames apart (a person coming back after a long gap is not a switch)."""
    hip = kp[:, list(HIPS), :2].mean(axis=1)
    idx = np.flatnonzero(np.isfinite(hip).all(axis=1))
    if len(idx) < 2:
        return 0
    close = np.diff(idx) <= MAX_STEP
    step = np.linalg.norm(np.diff(hip[idx], axis=0), axis=1)
    return int((close & (step > ID_JUMP * np.hypot(*wh))).sum())


def _edge(kp_frame: npt.NDArray[np.float32], wh: tuple[int, int]) -> float:
    x, y = kp_frame[list(HIPS), :2].mean(axis=0) / np.array(wh)
    return float(min(x, 1 - x, y, 1 - y))


def out_of_view(
    kp: npt.NDArray[np.float32],
    present: npt.NDArray[np.bool_],
    wh: tuple[int, int],
    gaps: pd.DataFrame,
) -> npt.NDArray[np.bool_]:
    """Frames where the person is off-screen, not missed: before they walk in and
    after they walk out at a border, and inside gaps that start at a border."""
    off = np.zeros(len(present), bool)
    margin = OCC["edge_margin"]
    for s, e, edge in gaps[["start", "end", "edge_dist"]].itertuples(index=False):
        if edge <= margin:
            off[s:e] = True
    hips_ok = np.isfinite(kp[:, list(HIPS), :2]).all(axis=(1, 2))
    seen = np.flatnonzero(stable_presence(present) & hips_ok)
    if len(seen):
        if _edge(kp[seen[0]], wh) <= margin:
            off[: seen[0]] = True
        if _edge(kp[seen[-1]], wh) <= margin:
            off[seen[-1] + 1 :] = True
    return off


def qc_clip(dataset: str, path: str, split: str, segs: pd.DataFrame) -> dict[str, Any]:
    z = np.load(skeleton_path(dataset, path))
    kp, wh = load_kp(z)
    present = z["detected"].astype(bool)
    n = len(present)
    counts: Counter[str] = Counter()
    y4 = to4(frame_labels(clean_segments(segs, n / FPS, counts), z["t_ms"]))

    gaps = find_disappearances(kp, present, wh, FPS)
    hidden = gaps[gaps["kind"] == "hidden_down"]
    in_hidden = np.zeros(n, bool)
    for s, e in hidden[["start", "end"]].itertuples(index=False):
        in_hidden[s:e] = True
    covered = present | in_hidden
    in_view = covered | out_of_view(kp, present, wh, gaps)
    fall = np.isin(y4, FALL)
    n_fall = int(fall.sum())

    def rate(mask: npt.NDArray[np.bool_]) -> float:
        return float(mask[fall].mean()) if n_fall else float("nan")

    jumps = id_jumps(kp, wh)
    reasons = []
    if in_view.mean() < MIN_COVERED:
        reasons.append("low_det")
    if n_fall and rate(covered) < MIN_COVERED_FALL:
        reasons.append("low_det_fall")
    if jumps > MAX_ID_JUMPS:
        reasons.append("id_switch")
    row = {
        "dataset": dataset,
        "path": path,
        "subject": int(z["subject"]),
        "cam": int(z["cam"]),
        "split": split,
        "n_frames": n,
        "duration_s": round(n / FPS, 3),
        "width": wh[0],
        "height": wh[1],
        "det_rate": float(present.mean()) if n else 0.0,
        "det_rate_fall": rate(present),
        "det_rate_covered": float(in_view.mean()) if n else 0.0,
        "det_rate_fall_covered": rate(covered),
        "hidden_frac_fall": rate(in_hidden),
        "n_hidden_gaps": len(hidden),
        "id_jumps": jumps,
        "qc_pass": not reasons,
        "qc_reason": ";".join(reasons),
    }
    # frames and detections per 4-class label, for the by-class plot
    for c, name in enumerate(CLASSES):
        row[f"_n_{name}"] = int((y4 == c).sum())
        row[f"_det_{name}"] = int(present[y4 == c].sum())
    return row


def qc_star(args: tuple[str, str, str, pd.DataFrame]) -> dict[str, Any]:
    return qc_clip(*args)


def clips(datasets: list[str]) -> list[tuple[str, str, str, pd.DataFrame]]:
    """(dataset, path, split, segments) for every .npz that has OmniFall labels."""
    out = []
    for ds in datasets:
        labels = load_labels(ds)
        segs = dict(iter(labels.groupby("path")))
        split = labels.groupby("path")["split"].first()
        on_disk = sorted(
            p.relative_to(DATA / "skeletons" / ds).with_suffix("").as_posix()
            for p in (DATA / "skeletons" / ds).rglob("*.npz")
        )
        missing = [p for p in on_disk if p not in split.index]
        if missing:
            print(f"{ds}: {len(missing)} skeletons without labels skipped")
        out += [(ds, p, str(split[p]), segs[p]) for p in on_disk if p in split.index]
    return out


def plot_by_dataset(index: pd.DataFrame) -> None:
    g = index.groupby("dataset")[["det_rate", "det_rate_covered"]].mean()
    ax = g.plot.bar(figsize=(8, 4), ylim=(0, 1), rot=0)
    ax.axhline(MIN_COVERED, color="grey", ls="--", lw=1)
    ax.set_ylabel("mean per clip")
    ax.set_title("Detection rate by dataset (covered = detected or hidden_down)")
    plt.tight_layout()
    plt.savefig(QC_DIR / "det_rate_by_dataset.png", dpi=120)
    plt.close()


def plot_by_class(full: pd.DataFrame) -> pd.DataFrame:
    """Frame-level detection rate per dataset and 4-class label."""
    n = full.groupby("dataset")[[f"_n_{c}" for c in CLASSES]].sum()
    det = full.groupby("dataset")[[f"_det_{c}" for c in CLASSES]].sum()
    rate = pd.DataFrame(
        det.to_numpy() / np.maximum(n.to_numpy(), 1), index=n.index, columns=CLASSES
    ).where(n.to_numpy() > 0)
    ax = rate.plot.bar(figsize=(9, 4), ylim=(0, 1), rot=0)
    ax.set_ylabel("detected frames / labelled frames")
    ax.set_title("Detection rate by class (raw)")
    plt.tight_layout()
    plt.savefig(QC_DIR / "det_rate_by_class.png", dpi=120)
    plt.close()
    return rate


def source_videos() -> dict[tuple[str, str], str]:
    """(dataset, path) -> video file, for the videos on this laptop."""
    inv = DATA / "inventory.csv"
    if not inv.exists():
        return {}
    df = pd.read_csv(inv, usecols=["dataset", "path", "file"])
    return {(d, p): f for d, p, f in df.itertuples(index=False) if Path(f).exists()}


def render_sample(dataset: str, path: str, out: Path, video: str | None) -> None:
    """Sample MP4 with the skeleton in its labelled class colour, drawn over the
    source video when `video` is given (to spot pose or label errors), else on
    black (what the sensor sees). Header: dataset and clip; footer: time and
    label, so you can check the label changes when the body does."""
    z = np.load(skeleton_path(dataset, path))
    n = len(z["detected"])
    segs = load_labels(dataset).query("path == @path")
    y4 = to4(frame_labels(clean_segments(segs, n / FPS), z["t_ms"]))
    w, h = int(z["width"]), int(z["height"])
    scale = 480 / max(w, h)
    view = (round(w * scale), round(h * scale))
    size = (view[0], view[1] + 2 * BAR)
    title = f"{dataset}  {path}" + ("" if video else "  (no video here)")
    # the same decoder and resampling as extraction, so frame i matches pose i
    frames = (f for _, f in iter_frames(video, FPS)) if video else None
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter.fourcc(*"mp4v"), FPS, size)
    for kp, det, t_ms, y in zip(z["kp"], z["detected"], z["t_ms"], y4):
        img = np.zeros((size[1], size[0], 3), np.uint8)
        frame = next(frames, None) if frames is not None else None
        if frame is not None:
            img[BAR : BAR + view[1]] = cv2.resize(frame, view) // 4 * 3  # dimmed
        name, colour = CLASS_STYLE.get(int(y), UNLABELLED_STYLE)
        if det:
            pts = np.c_[kp[:, 0] * size[0], kp[:, 1] * (size[1] - 2 * BAR) + BAR]
            pts = pts.astype(int)
            for a, b in BONES:
                cv2.line(img, tuple(pts[a]), tuple(pts[b]), colour, 2)
            for p in pts[KEYPOINTS]:
                cv2.circle(img, tuple(p), 3, (255, 255, 255), -1)
        else:
            cv2.putText(img, "no pose", (10, BAR + 30), 0, 0.8, (0, 0, 255), 2)
        cv2.rectangle(img, (0, 0), (size[0], BAR), (40, 40, 40), -1)
        cv2.putText(img, title, (6, 20), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, size[1] - BAR), (size[0], size[1]), (40, 40, 40), -1)
        foot = f"{t_ms / 1000:6.2f} s   {name}"
        cv2.putText(img, foot, (6, size[1] - 9), 0, 0.6, colour, 2, cv2.LINE_AA)
        vw.write(img)
    vw.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--out", type=Path, default=DATA / "skeletons" / "index.csv")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument(
        "--black", action="store_true", help="samples on black, not the video"
    )
    ap.add_argument(
        "--clip",
        nargs="+",
        metavar="DATASET:PATH",
        help="only render these clips (e.g. of-syn:fall/fall_mi_062), no QC",
    )
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    if args.clip:
        pairs = [tuple(c.split(":", 1)) for c in args.clip]
        render_samples(pd.DataFrame(pairs, columns=["dataset", "path"]), args.black)
        return

    todo = clips(args.datasets)
    print(f"QC on {len(todo)} clips with {args.workers} workers")
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(qc_star, todo, chunksize=16))
    full = pd.DataFrame(rows).sort_values(["dataset", "path"], kind="stable")
    full = full.reset_index(drop=True)
    index = full[INDEX_COLUMNS]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(index)} clips)")

    QC_DIR.mkdir(parents=True, exist_ok=True)
    plot_by_dataset(index)
    by_class = plot_by_class(full)
    trainval = index["split"].isin(["train", "val"])
    excluded = index[trainval & ~index["qc_pass"]]
    excluded[["dataset", "path", "split", "det_rate_covered", "qc_reason"]].to_csv(
        QC_DIR / "excluded.csv", index=False
    )

    summary = index.groupby(["dataset", "split"]).agg(
        clips=("path", "size"),
        qc_pass=("qc_pass", "sum"),
        det_rate=("det_rate", "mean"),
        det_rate_covered=("det_rate_covered", "mean"),
        hidden_gaps=("n_hidden_gaps", "sum"),
    )
    print(summary.round(3).to_string())
    print("\ndetection rate by class:\n" + by_class.round(3).to_string())
    print(f"\n{len(excluded)} train/val clips excluded:")
    print(excluded["qc_reason"].str.split(";").explode().value_counts().to_string())

    if args.samples:
        # one clip per dataset first, then random clips up to --samples
        seed = CFG["seed"]
        first = index.groupby("dataset").sample(1, random_state=seed)
        rest = index.drop(first.index).sample(frac=1, random_state=seed)
        render_samples(pd.concat([first, rest]).head(args.samples), args.black)


def render_samples(pick: pd.DataFrame, black: bool) -> None:
    videos = {} if black else source_videos()
    for r in pick.itertuples():
        name = f"{r.dataset}_{str(r.path).replace('/', '_')}.mp4"
        video = videos.get((r.dataset, r.path))
        render_sample(r.dataset, r.path, QC_DIR / "samples" / name, video)
    on_video = sum((r.dataset, r.path) in videos for r in pick.itertuples())
    print(f"{len(pick)} sample videos ({on_video} on video) in {QC_DIR / 'samples'}")


if __name__ == "__main__":
    main()
