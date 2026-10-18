"""Train a window classifier on Contract 3 windows and select the best epoch on val.

    python scripts/06_train.py --model gru --no-augment --seed 1 --out models/e1_gru_s1
    python scripts/06_train.py --model gru --train-sets GMDCSA24 caucafall le2i mcfd ...
    python scripts/06_train.py --windows-dir data/windows_dryrun --epochs 3 ...  # dry run

Writes to --out: model.pt, feat_stats.npz, config.yaml (config + run args + git
commit + laptop), train_log.csv (rewritten every epoch), val_logits.npy (best
epoch, all of val in val.npz order), confusion.png.

The best epoch is selected on val windows from the --train-sets datasets only (a
model trained on the core sets is not judged on OF-Syn val); AP and F1 on all of
val are logged alongside as val_all_*. --select event picks the epoch with the
fewest false FALL alerts/h at FALL sensitivity >= --select-target on those val
clips (the decision metric), with window AP as the tie-break; --select ap (the
default, E1-E7) uses window AP then macro F1.
"""

import argparse
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fallsense import calibrate, dataset, metrics, windows
from fallsense.config import DATA, load_config
from fallsense.models import MODELS, build_model

CFG = load_config()
FALLING = windows.FALLING
CONTROLLED_LOW = 3
PATIENCE = 8
SELECT_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def keep_datasets(
    arrays: dict[str, np.ndarray], index: pd.DataFrame, sets: list[str]
) -> dict[str, np.ndarray]:
    """Only windows whose clip (row of index.csv) belongs to one of `sets`."""
    ok = in_datasets(arrays, index, sets)
    return {k: v[ok] for k, v in arrays.items()}


def in_datasets(
    arrays: dict[str, np.ndarray], index: pd.DataFrame, sets: list[str]
) -> np.ndarray:
    """Boolean mask: window's clip belongs to one of `sets`."""
    return index["dataset"].isin(sets).to_numpy()[arrays["clip_id"]]


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


def class_weights(y: np.ndarray, low_weight: float = 1.0) -> torch.Tensor:
    """w_c = sqrt(N / (4 n_c)), renormalised to mean 1; controlled_low x low_weight."""
    n_c = np.maximum(np.bincount(y, minlength=windows.N_CLASSES), 1)
    w = np.sqrt(len(y) / (windows.N_CLASSES * n_c))
    w = w / w.mean()
    w[CONTROLLED_LOW] *= low_weight
    return torch.tensor(w, dtype=torch.float32)


def feature_stats(feats: np.ndarray) -> dict[str, np.ndarray]:
    """Per-feature mean and std over every frame of every train window."""
    flat = feats.reshape(-1, feats.shape[-1])
    sd = flat.std(axis=0)
    return {
        "mu": flat.mean(axis=0).astype(np.float32),
        "sd": np.where(sd < 1e-6, 1.0, sd).astype(np.float32),
    }


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch: int = 1024) -> np.ndarray:
    model.eval()
    out = [
        model(torch.from_numpy(x[i : i + batch])).numpy()
        for i in range(0, len(x), batch)
    ]
    return np.concatenate(out) if out else np.empty((0, windows.N_CLASSES), np.float32)


def val_metrics(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(falling-vs-rest average precision, 4-class macro F1)."""
    p = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    is_falling = y == FALLING
    ap = average_precision_score(is_falling, p[:, FALLING]) if is_falling.any() else 0.0
    f1 = f1_score(
        y,
        p.argmax(1),
        average="macro",
        labels=range(windows.N_CLASSES),
        zero_division=0,
    )
    return float(ap), float(f1)


def event_score(
    logits: np.ndarray,
    val: dict[str, np.ndarray],
    clips: pd.DataFrame,
    events: pd.DataFrame,
    target: float,
) -> tuple[float, float, float]:
    """(fewest false FALL/h at FALL sensitivity >= target, its sensitivity, threshold).

    inf false/h when no threshold reaches the target.
    """
    sc = calibrate.scores_frame(val, calibrate.softmax_t(logits))
    sc = sc[sc["clip_id"].isin(set(clips["clip_id"]))]
    best = (np.inf, 0.0, np.nan)
    for t in SELECT_GRID:
        alerts = calibrate.alerts_from_scores(sc, {"t_high": t, "t_low": t})
        r = metrics.summarise(*metrics.match_events(clips, events, alerts))
        if r["sensitivity_alarm"] >= target and r["false_alarms_per_h"] < best[0]:
            best = (r["false_alarms_per_h"], r["sensitivity_alarm"], float(t))
    return best


def save_confusion(logits: np.ndarray, y: np.ndarray, path: Path) -> None:
    cm = confusion_matrix(y, logits.argmax(1), labels=range(windows.N_CLASSES))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for (r, c), n in np.ndenumerate(cm):
        ax.text(c, r, str(n), ha="center", va="center")
    ax.set_xticks(range(windows.N_CLASSES), CFG["classes"], rotation=30)
    ax.set_yticks(range(windows.N_CLASSES), CFG["classes"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("val confusion (best epoch)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=sorted(MODELS), default="gru")
    ap.add_argument(
        "--train-sets",
        nargs="+",
        help="datasets to train on (default: all in train.npz)",
    )
    ap.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--occlude", action="store_true", help="simulated occlusion (E7; needs M19)"
    )
    ap.add_argument("--seed", type=int, default=CFG["seed"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--windows-dir", type=Path, default=DATA / "windows")
    ap.add_argument("--index", type=Path, default=DATA / "skeletons" / "index.csv")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument(
        "--low-weight",
        type=float,
        default=1.0,
        help="extra loss weight on controlled_low (near-miss) windows",
    )
    ap.add_argument(
        "--select",
        choices=["ap", "event"],
        default="ap",
        help="epoch selection: window AP, or false FALL/h at --select-target",
    )
    ap.add_argument("--select-target", type=float, default=0.80)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--threads",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="torch threads (cap it when several runs share a laptop)",
    )
    args = ap.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)

    train_npz = args.windows_dir / "train.npz"
    train = windows.load_windows("train", path=train_npz)
    val = windows.load_windows("val", path=args.windows_dir / "val.npz")
    index = None
    if args.train_sets or args.occlude or args.select == "event":
        index = pd.read_csv(args.index)
    if args.train_sets:
        assert index is not None
        train = keep_datasets(train, index, args.train_sets)
    # select epochs on val windows from the training datasets only
    sel = (
        in_datasets(val, index, args.train_sets)
        if args.train_sets and index is not None
        else np.ones(len(val["y"]), bool)
    )
    print(
        f"train {len(train['y'])} windows, val {len(val['y'])} windows "
        f"({int(sel.sum())} from the train sets, used for selection)"
    )
    print("train classes:", np.bincount(train["y"], minlength=windows.N_CLASSES))

    # z-score stats over train windows without augmentation
    stats = feature_stats(dataset.window_features(train))
    np.savez(args.out / "feat_stats.npz", **stats)
    x_val = (dataset.window_features(val) - stats["mu"]) / stats["sd"]
    x_val = x_val.astype(np.float32)

    bank = None
    if args.occlude:
        assert index is not None
        bank = dataset.ClipBank.from_index(index, train["clip_id"])
    if args.train_sets:
        # WindowDataset reads an npz; the subset only lives until it is loaded
        with tempfile.TemporaryDirectory() as tmp:
            train_npz = Path(tmp) / "train.npz"
            np.savez(train_npz, **train)
            ds = dataset.WindowDataset(
                train_npz, stats, augment=args.augment, seed=args.seed, bank=bank
            )
    else:
        ds = dataset.WindowDataset(
            train_npz, stats, augment=args.augment, seed=args.seed, bank=bank
        )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        worker_init_fn=dataset.worker_init_fn,
        generator=torch.Generator().manual_seed(args.seed),
    )

    model = build_model(args.model)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights(train["y"], args.low_weight), label_smoothing=0.05
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sel_clips = sel_events = None
    if args.select == "event":
        assert index is not None
        sel_clips = metrics.eval_clips(index, "val")
        if args.train_sets:
            sel_clips = sel_clips[sel_clips["dataset"].isin(args.train_sets)]
        sel_events = metrics.fall_events(sel_clips)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    with open(args.out / "config.yaml", "w", encoding="utf-8") as f:
        run = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        run["git_commit"] = git_commit()
        run["laptop"] = platform.node()
        run["val_selection"] = " ".join(args.train_sets or ["all"])
        yaml.safe_dump({"run": run, "config": CFG}, f, sort_keys=False)

    log, best, best_logits, stale = [], (-np.inf, -1.0), None, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
        sched.step()

        logits = predict(model, x_val)
        ap_fall, f1 = val_metrics(logits[sel], val["y"][sel])
        ap_all, f1_all = val_metrics(logits, val["y"])
        row = {"epoch": epoch, "loss": total / n, "val_ap_falling": ap_fall,
               "val_macro_f1": f1, "val_all_ap_falling": ap_all,
               "val_all_macro_f1": f1_all}  # fmt: skip
        extra = ""
        if args.select == "event":
            fph, sens, t_best = event_score(
                logits, val, sel_clips, sel_events, args.select_target
            )
            row.update(val_false_per_h=fph, val_sens=sens, val_t=t_best)
            key = (-fph, ap_fall)
            extra = f"  false/h {fph:.1f} at sens {sens:.3f} (t {t_best})"
        else:
            key = (ap_fall, f1)
        secs = time.perf_counter() - t0
        row["seconds"] = secs
        log.append(row)
        pd.DataFrame(log).to_csv(args.out / "train_log.csv", index=False)
        improved = key > best
        print(
            f"epoch {epoch:2d}  loss {total / n:.4f}  val AP(falling) {ap_fall:.4f}"
            f"  macro F1 {f1:.4f}  (all val {ap_all:.4f} / {f1_all:.4f})"
            f"{extra}  {secs:.1f}s" + ("  *" if improved else "")
        )
        if epoch == 1:
            # epoch 1 includes DataLoader worker start-up (slow on Windows)
            print(
                f"{secs:.1f} s for epoch 1 (incl. worker start-up), "
                f"ETA up to {secs * args.epochs / 60:.1f} min"
            )
            if secs > 120:
                print("WARNING: over 2 min per epoch; precompute train_aug.npz")
        if improved:
            best, best_logits, stale = key, logits, 0
            torch.save(
                {"model": args.model, "state_dict": model.state_dict()},
                args.out / "model.pt",
            )
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"early stop: {PATIENCE} epochs without improvement")
                break

    np.save(args.out / "val_logits.npy", best_logits)
    save_confusion(best_logits, val["y"], args.out / "confusion.png")
    if args.select == "event":
        print(
            f"best val AP(falling) {best[1]:.4f}, false FALL/h {-best[0]:.1f} at "
            f"sens >= {args.select_target:.0%} -> {args.out}"
        )
    else:
        print(
            f"best val AP(falling) {best[0]:.4f}, macro F1 {best[1]:.4f} -> {args.out}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
