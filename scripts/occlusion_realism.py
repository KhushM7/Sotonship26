"""Occlusion realism check and disappearance threshold check -> reports/occlusion_realism.md.

    python scripts/occlusion_realism.py

1. Real hidden falls: every OCCU disappearance that starts in a falling/fallen frame.
2. Simulated hidden falls: visible falls from the other train sets passed through
   `simulate_occlusion` (onset inside the fall), then the same `find_disappearances`.
3. Simulated normal activity: the same for non-fall moments, to see how often the
   classifier calls a person walking behind furniture `hidden_down`.
4. Threshold check: `left` / `hidden_down` gaps per dataset on real train clips, and
   how many `hidden_down` gaps fall outside labelled falls (future false CHECKs).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fallsense.config import DATA, load_config
from fallsense.labels import clean_segments, frame_labels, load_labels, to4
from fallsense.occlusion import (
    ANKLES,
    HIPS,
    NOSE,
    OCC,
    VISIBLE,
    find_disappearances,
    simulate_occlusion,
)

CFG = load_config()
KEYPOINTS = CFG["keypoints"]
FPS = CFG["fps"]
SIM_SETS = ["le2i", "GMDCSA24", "caucafall", "mcfd", "of-syn"]
GROUPS = {"head": [NOSE], "hips": list(HIPS), "ankles": list(ANKLES)}
SEED = CFG["seed"]
SWEEP = [0.15, 0.2, 0.3, 0.4, 0.5]  # descent_speed values shown in the report
OUT = Path(__file__).resolve().parents[1] / "reports" / "occlusion_realism.md"


def load_clip(dataset: str, path: str) -> dict | None:
    f = DATA / "skeletons" / dataset / f"{path}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    wh = (int(z["width"]), int(z["height"]))
    kp = z["kp"][:, KEYPOINTS][:, :, [0, 1, 3]].astype(np.float32)
    kp[..., 0] *= wh[0]
    kp[..., 1] *= wh[1]
    present = z["detected"].astype(bool)
    dur = len(present) / FPS
    segs = clean_segments(load_labels(dataset).query("path == @path"), dur)
    y4 = to4(frame_labels(segs, z["t_ms"]))
    return {"kp": kp, "present": present, "wh": wh, "y4": y4, "segs": segs}


def loss_order(kp: np.ndarray, start: int) -> dict[str, float]:
    """Frames before `start` at which each group's visibility dropped for good."""
    out = {}
    for name, idx in GROUPS.items():
        vis = np.nanmean(kp[max(0, start - 30) : start, idx, 2], axis=1)
        low = ~(vis > VISIBLE)
        # length of the low-visibility run that ends at the gap
        run = 0
        for v in low[::-1]:
            if not v:
                break
            run += 1
        out[f"{name}_low_frames"] = float(run)
    return out


def gaps_with_context(c: dict, after: int = 0) -> pd.DataFrame:
    gaps = find_disappearances(c["kp"], c["present"], c["wh"], FPS)
    gaps = gaps[gaps["start"] >= after]
    if gaps.empty:
        return gaps
    rows = [loss_order(c["kp"], int(s)) for s in gaps["start"]]
    gaps = pd.concat([gaps.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    gaps["during_fall"] = [
        bool(np.isin(c["y4"][s : s + 3], [1, 2]).any()) for s in gaps["start"]
    ]
    return gaps


def md_table(df: pd.DataFrame) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    head = [str(df.index.name or ""), *map(str, df.columns)]
    rows = [[str(i), *map(str, r)] for i, r in zip(df.index, df.to_numpy())]
    lines = [head, ["---"] * len(head), *rows]
    return "\n".join("| " + " | ".join(r) + " |" for r in lines)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "edge_dist",
        "descent_speed",
        "duration_s",
        "ankles_low_frames",
        "hips_low_frames",
        "head_low_frames",
    ]
    q = df[cols].quantile([0.25, 0.5, 0.75]).T
    q.columns = ["p25", "median", "p75"]
    return q.round(2)


def threshold_sweep(gaps: pd.DataFrame) -> pd.DataFrame:
    """OCCU hidden falls caught vs false hidden_down outside falls, per threshold."""
    mid = gaps["edge_dist"] > OCC["edge_margin"]
    occu_falls = gaps["during_fall"] & (gaps["dataset"] == "occu")
    rows = []
    for thr in SWEEP:
        hd = mid & (gaps["descent_speed"] > thr)
        rows.append(
            {
                "descent_speed": thr,
                "occu_hidden_falls_caught": f"{(hd & occu_falls).sum() / occu_falls.sum():.0%}",
                "false_hidden_down_outside_falls": int(
                    (hd & ~gaps["during_fall"]).sum()
                ),
                "of_which_mcfd": int(
                    (hd & ~gaps["during_fall"] & (gaps["dataset"] == "mcfd")).sum()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("descent_speed")


STATUS = [
    "## Status: accepted for now, revisit after E7",
    "",
    (
        "Decision (17 Oct 2026): the simulation is accepted as it is and judged by "
        "experiment E7, not by further tuning."
    ),
    "",
    (
        "- Matches real OCCU: distance from the frame edge at loss, loss order (ankles "
        "first; hips and head visible until the person vanishes), and few false "
        "`hidden_down` during simulated normal activity."
    ),
    (
        "- Known gap: simulated hidden falls vanish at a lower hip speed than real OCCU "
        "ones, so fewer of them are classified `hidden_down` (see the tables above). "
        "Simulated losses happen within about 0.3 s of the peak descent, so the gap is "
        "in the measured speed at loss, not the timing."
    ),
    (
        "- Revisit if E7 shows no gain on OCCU val/test or the staged hidden-fall clips, "
        "or more false alerts elsewhere. Starting points: `RISE_EDGE_IN_DROP`, "
        "`MIN_DROP` and the `hips_hidden` loss rule in `src/fallsense/occlusion.py`."
    ),
    "",
    "## E7 outcome",
    "",
    (
        "_To be filled in by Dev B when E7 runs (see the E7 row in the plans): val "
        "results with and without simulated occlusion on all of val, OCCU val, and false "
        "alerts per hour elsewhere; the run names; and whether E7 is kept. Copy the same "
        "numbers into `reports/experiments.csv`._"
    ),
    "",
]


def main() -> None:
    rng = np.random.default_rng(SEED)
    real, real_all, sim_fall, sim_other, per_dataset = [], [], [], [], []
    for ds in ["occu", *SIM_SETS]:
        labels = load_labels(ds)
        train_paths = labels.loc[labels["split"] == "train", "path"].unique()
        for path in train_paths:
            c = load_clip(ds, path)
            if c is None:
                continue
            g = gaps_with_context(c)
            per_dataset.append(
                {
                    "dataset": ds,
                    "left": int((g.get("kind") == "left").sum()) if len(g) else 0,
                    "hidden_down": int((g.get("kind") == "hidden_down").sum())
                    if len(g)
                    else 0,
                    "hidden_down_outside_falls": int(
                        ((g.get("kind") == "hidden_down") & ~g["during_fall"]).sum()
                    )
                    if len(g)
                    else 0,
                }
            )
            if len(g):
                real_all.append(g.assign(dataset=ds))
            if ds == "occu":
                if len(g):
                    real.append(g[g["during_fall"]].assign(dataset=ds))
                continue
            falls = c["segs"][c["segs"]["label"] == 1]
            for s in falls.itertuples():
                # A static occluder hides someone as they go down: onset in the fall's first third.
                onset = int(rng.uniform(s.start, s.start + (s.end - s.start) / 3) * FPS)
                kp, pres = simulate_occlusion(
                    c["kp"],
                    c["present"],
                    c["wh"],
                    rng,
                    onset,
                    rng.choice(["rise", "box"]),
                )
                g2 = gaps_with_context(dict(c, kp=kp, present=pres), after=onset)
                if len(g2):
                    sim_fall.append(g2.head(1).assign(dataset=ds))
            upright = np.flatnonzero(c["y4"] == 0)
            if len(upright) > 30 and rng.random() < 0.5:
                onset = int(rng.choice(upright[:-15]))
                kp, pres = simulate_occlusion(
                    c["kp"],
                    c["present"],
                    c["wh"],
                    rng,
                    onset,
                    rng.choice(["rise", "box"]),
                )
                g3 = gaps_with_context(dict(c, kp=kp, present=pres), after=onset)
                if len(g3):
                    sim_other.append(g3.head(1).assign(dataset=ds))

    real_df = pd.concat(real, ignore_index=True)
    fall_df = pd.concat(sim_fall, ignore_index=True)
    other_df = pd.concat(sim_other, ignore_index=True)
    counts = pd.DataFrame(per_dataset).groupby("dataset").sum(numeric_only=True)

    def share(df: pd.DataFrame) -> str:
        return f"{(df['kind'] == 'hidden_down').mean():.0%} classified hidden_down"

    sweep = threshold_sweep(pd.concat(real_all, ignore_index=True))

    lines = [
        "# Occlusion realism check",
        "",
        (
            "Generated by `scripts/occlusion_realism.py` (seed "
            f"{SEED}; edge_margin {OCC['edge_margin']}, "
            f"descent_speed {OCC['descent_speed']}, "
            f"min_visible_kp {OCC['min_visible_kp']}). Train clips only."
        ),
        "",
        (
            "- Real hidden falls (OCCU, gap starting in a fall/fallen frame): "
            f"{len(real_df)} gaps, {share(real_df)}"
        ),
        (
            "- Simulated hidden falls (visible falls from "
            f"{', '.join(sorted(fall_df['dataset'].unique()))}): "
            f"{len(fall_df)} gaps, {share(fall_df)}"
        ),
        (
            f"- Simulated occlusion during normal activity: {len(other_df)} gaps, "
            f"{share(other_df)} (lower is better: these would be false `hidden_down`)"
        ),
        "",
        (
            "`*_low_frames`: how many frames before the person vanished that body "
            "part's visibility was already below 0.5 (loss order: larger = lost earlier)."
        ),
        "",
        "## Real hidden falls (OCCU)",
        "",
        md_table(summary(real_df)),
        "",
        "## Simulated hidden falls",
        "",
        md_table(summary(fall_df)),
        "",
        "## Simulated occlusion during normal activity",
        "",
        md_table(summary(other_df)),
        "",
        "## Disappearances on real train clips with the current thresholds",
        "",
        md_table(counts),
        "",
        "## Choosing descent_speed (body heights/s, real train clips)",
        "",
        (
            "Hip speed is measured in body heights/s (body height = the tallest the person "
            "stood in the last 2 s), because in frame heights/s the same fall reads 3x "
            "faster on a close camera than a far one (visible-fall speed 0.15 on MCFD vs "
            "0.47 on GMDCSA24 in frame heights/s; 0.67-0.99 in body heights/s)."
        ),
        "",
        md_table(sweep),
        "",
        *STATUS,
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
