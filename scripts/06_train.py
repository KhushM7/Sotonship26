"""Train a window classifier on Contract 3 windows and select the best epoch on val.

    python scripts/06_train.py --model gru --no-augment --seed 1 --out models/e1_gru_s1
    python scripts/06_train.py --model gru --train-sets GMDCSA24 caucafall le2i mcfd ...
    python scripts/06_train.py --windows-dir data/windows_dryrun --epochs 3 ...  # dry run

Writes to --out: model.pt, feat_stats.npz, config.yaml (config + run args),
train_log.csv, val_logits.npy (best epoch, val.npz order), confusion.png.
"""

import argparse
import os
import random
import sys
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

from fallsense import dataset, windows
from fallsense.config import DATA, load_config
from fallsense.models import MODELS, build_model

CFG = load_config()
FALLING = windows.FALLING
PATIENCE = 8


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def keep_datasets(
    arrays: dict[str, np.ndarray], index: pd.DataFrame, sets: list[str]
) -> dict[str, np.ndarray]:
    """Only windows whose clip (row of index.csv) belongs to one of `sets`."""
    ok = index["dataset"].isin(sets).to_numpy()[arrays["clip_id"]]
    return {k: v[ok] for k, v in arrays.items()}


def class_weights(y: np.ndarray) -> torch.Tensor:
    """w_c = sqrt(N / (4 n_c)), renormalised to mean 1."""
    n_c = np.maximum(np.bincount(y, minlength=windows.N_CLASSES), 1)
    w = np.sqrt(len(y) / (windows.N_CLASSES * n_c))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


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
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    args.out.mkdir(parents=True, exist_ok=True)

    train_npz = args.windows_dir / "train.npz"
    train = windows.load_windows("train", path=train_npz)
    val = windows.load_windows("val", path=args.windows_dir / "val.npz")
    index = None
    if args.train_sets or args.occlude:
        index = pd.read_csv(args.index)
    if args.train_sets:
        assert index is not None
        train = keep_datasets(train, index, args.train_sets)
        train_npz = args.out / "train_subset.npz"
        np.savez(train_npz, **train)
    print(f"train {len(train['y'])} windows, val {len(val['y'])} windows")
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
        weight=class_weights(train["y"]), label_smoothing=0.05
    )
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    with open(args.out / "config.yaml", "w", encoding="utf-8") as f:
        run = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        yaml.safe_dump({"run": run, "config": CFG}, f, sort_keys=False)

    log, best, best_logits, stale = [], (-1.0, -1.0), None, 0
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
        ap_fall, f1 = val_metrics(logits, val["y"])
        secs = time.perf_counter() - t0
        log.append(
            {"epoch": epoch, "loss": total / n, "val_ap_falling": ap_fall,
             "val_macro_f1": f1, "seconds": secs}
        )  # fmt: skip
        improved = (ap_fall, f1) > best
        print(
            f"epoch {epoch:2d}  loss {total / n:.4f}  val AP(falling) {ap_fall:.4f}"
            f"  macro F1 {f1:.4f}  {secs:.1f}s" + ("  *" if improved else "")
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
            best, best_logits, stale = (ap_fall, f1), logits, 0
            torch.save(
                {"model": args.model, "state_dict": model.state_dict()},
                args.out / "model.pt",
            )
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"early stop: {PATIENCE} epochs without improvement")
                break

    pd.DataFrame(log).to_csv(args.out / "train_log.csv", index=False)
    np.save(args.out / "val_logits.npy", best_logits)
    save_confusion(best_logits, val["y"], args.out / "confusion.png")
    print(f"best val AP(falling) {best[0]:.4f}, macro F1 {best[1]:.4f} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
