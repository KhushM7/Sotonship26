# Training Run Plan – who trains what, where

Prep for after Dev B's model code is merged (M9 `models.py` / `06_train.py`, M10 `calibrate.py`) and S7/S11 are done. It decides which laptop runs which training jobs, gives the run queue with commands, and fixes the rules that keep results comparable. Fill in the measured numbers as they arrive; nothing here changes the experiments themselves (see the master plan, Phase 5–6).

## 1. The two machines

| | Laptop A (Dev A) | Laptop B (Dev B) |
| --- | --- | --- |
| CPU | Intel Core Ultra 9 285H, 16 cores / 16 threads (6P + 8E + 2 LP-E) | Intel Core i5-12450H, 8 cores / 12 threads (4P + 4E) |
| RAM | 63 GB | 16 GB (about 1/4 of A; Windows and a browser take 5–7 GB) |
| Free disk | ~276 GB | 30 GB cap agreed for the project |
| Skeletons on disk | all 7 datasets (`C:\fall-data\skeletons`, 6,802 clips) | own 3 + A's 4 zips after S7/S11 |
| Source videos | Le2i, OCCU, OF-Syn, OOPS | GMDCSA24, CAUCAFall, MCFD |
| Best at | long full runs, several runs at once | writing and debugging training code |

What costs time in training (CPU only, no GPU):

1. **Model maths** – small GRU/TCN; scales with cores, but poorly past ~8 threads for a model this small.
2. **Data loading and augmentation** – Python per window (`dataset.py`, `ClipBank` simulated occlusion for E7). Scales with DataLoader workers, i.e. cores. Probably the bottleneck for E2–E7.
3. **Memory** – windows for all train sets, plus the clip bank for E7. If a laptop starts swapping, every run slows several times over.

**Expectation before measuring.** A has twice the cores and four times the RAM, so a single run should be roughly 1.5–2× faster on A (small models don't scale perfectly with cores), and A can run 2–3 runs at once where B can run one. With about 9 GB usable, B is fine for the core sets (E1–E4) but tight for E5–E7, which add OF-Syn's windows and, for E7, the clip bank for simulated occlusion. So expect **Mode 3** below, with the heavy rows on A, unless B's benchmark is fast and stays well under 9 GB.

**Settings for B:** `--workers 4` (the default) and the default thread count (11) for single runs; never two runs at once on B.

## 2. Step 0 – measure, then decide (30 min, once)

Both laptops, same commit on `main`, same `requirements.lock`:

```powershell
git pull --ff-only
python scripts\05_build_windows.py                        # once per laptop; ~minutes
Get-Content reports\windows_summary.csv                   # both laptops must match exactly
python scripts\06_train.py --model gru --seed 1 --epochs 2 --out models\bench_gru
python scripts\06_train.py --model gru --seed 1 --epochs 2 --occlude --out models\bench_gru_occ
```

Record per laptop: seconds per epoch (printed after epoch 1, also in `train_log.csv`), peak RAM (Task Manager), and whether the disk light shows swapping.

| Laptop | Train windows | s/epoch (plain) | s/epoch (`--occlude`) | Peak RAM | Swapping? |
| --- | --- | --- | --- | --- | --- |
| A | | | | | |
| B | | | | | |

**Decision rule**

| Measured on B | Mode |
| --- | --- |
| < 60 s/epoch with `--occlude`, peak RAM under 9 GB | **Mode 1 – B trains everything.** Simplest; A builds the evaluator, baseline and tracker meanwhile |
| 60–180 s/epoch, or peak RAM 9–13 GB | **Mode 3 – split rows across both laptops** (section 4) |
| > 180 s/epoch, peak RAM over 13 GB, or swapping | **Mode 2 – A runs every full training;** B develops on dry runs only |

Run the benchmark on B with the full train set (no `--train-sets`) so it shows the worst-case RAM, not the core sets alone.

One run ≈ s/epoch × ~25 epochs (early stop, ≤ 40). The full queue (section 3) is ~25 runs, so 60 s/epoch ≈ 10 h of runs on one laptop – too long for one machine in a hackathon, hence the split.

## 3. Run queue

Every E-row has 3 seeds (1, 2, 3). Run names: `e<row>_<model>_s<seed>` (for example `e2_gru_s1`). "Core" = `GMDCSA24 caucafall le2i mcfd`.

| Row | Depends on | Command (seed 1 shown) | Runs |
| --- | --- | --- | --- |
| E0 | – | rule baseline, Dev A (`baseline.py`); no training | 1 |
| E1 | windows | `06_train.py --model gru --no-augment --train-sets GMDCSA24 caucafall le2i mcfd --seed 1 --out models\e1_gru_s1` | 3 |
| E2 | windows | `06_train.py --model gru --augment --train-sets GMDCSA24 caucafall le2i mcfd --seed 1 --out models\e2_gru_s1` | 3 |
| E3 | windows | `06_train.py --model tcn --augment --train-sets GMDCSA24 caucafall le2i mcfd --seed 1 --out models\e3_tcn_s1` | 3 |
| E4 | best of E2/E3 | `06_train.py --model <best> --augment --train-sets GMDCSA24 caucafall le2i mcfd occu --seed 1 --out models\e4_<best>_s1` | 3 |
| E5 | best of E2/E3 | `06_train.py --model <best> --augment --train-sets GMDCSA24 caucafall le2i mcfd occu of-syn --seed 1 --out models\e5_<best>_s1` | 3 |
| E6 | best of E4/E5 | as the best row, `--train-sets` without `caucafall` | 3 |
| E7 | best of E4/E5 | as the best row plus `--occlude` | 3 |
| E8 | E1–E7 val results | best config on the care-home subset vs all data (deferred; see `reports/data_scope_review.md`) | 3–6 |

Dependency waves – everything inside a wave can run at the same time:

1. **Wave 1:** E1, E2, E3 (9 runs, core sets only – the smallest and fastest).
2. **Wave 2:** E4, E5 (6 runs; E5 is the big one, with OF-Syn).
3. **Wave 3:** E6, E7 (6 runs; E7 is the slowest, with simulated occlusion).
4. **Wave 4:** E8 (later), then final calibration of the chosen run.

After each run: `python -c "from fallsense.calibrate import write_scores; write_scores('models/<run>', r'$env:FALL_DATA\windows\val.npz')"` writes `reports/scores/<run>/val.parquet` (Contract 4) for the evaluator. Test, staged_test and OOPS scores are written **once**, for the final frozen model only.

## 4. Mode 3 – splitting across both laptops

- **Split by row, never by seed.** All 3 seeds of a row run on the same laptop, so seed spread is not mixed with machine differences.
- **Wave 1:** E1 on B, E2 + E3 on A (A runs 2–3 jobs at once, see below).
- **Wave 2:** E4 on B, E5 on A (E5 has the most windows; A has the RAM).
- **Wave 3:** E6 on B, E7 on A (E7 is the most CPU-heavy).
- **Wave 4:** E8 and the final run on A.

**Several runs at once on laptop A.** One small-model run does not use 16 cores well, so 2–3 runs side by side give more total throughput. Each run needs its own thread cap, or they fight over the cores:
- **Prep for Dev B:** add `--threads` to `06_train.py` (default: the current `cpu_count - 1`) so `torch.set_num_threads(args.threads)` can be capped. About 5 minutes of work.
- On A then: 3 runs × `--threads 4 --workers 3` (~21 of 16 cores is fine, the workers mostly wait on Python). Check that the s/epoch per run is no worse than ~1.5× the single-run number; if it is, drop to 2 at a time.

Start each run in its own terminal so one crash doesn't stop the others, and log to a file:

```powershell
python scripts\06_train.py ... --threads 4 --workers 3 --out models\e2_gru_s1 2>&1 | Tee-Object models\e2_gru_s1.log
```

## 5. Rules that keep results comparable

1. **Same code:** every run comes from a commit on `main`. Record `git rev-parse --short HEAD` and the laptop (A/B) in `reports/experiments.csv` next to the run name.
2. **Same environment:** both laptops install from `requirements.lock`; check that `python -c "import torch; print(torch.__version__)"` matches.
3. **Same windows:** both laptops build windows from the same `index.csv`, and `reports\windows_summary.csv` must be identical. If not, stop and find out why before training.
4. **One owner of results:** Dev B chooses the runs and fills in `reports/experiments.csv`, whichever laptop ran them.
5. **Select on val only.** Test, staged_test and OOPS are opened once, for the final model.

## 6. Moving runs between laptops

A run folder is a few MB: `model.pt`, `feat_stats.npz`, `config.yaml`, `train_log.csv`, `val_logits.npy`, `confusion.png`, plus `reports/scores/<run>/val.parquet`.

- Zip `models\<run>` and `reports\scores\<run>` together and upload them to the Drive folder `fallsense-data/runs/`.
- The other laptop unzips into the same paths. `models/` and `reports/scores/` stay out of git (check `.gitignore` before the first upload); only `reports/experiments.csv` and the eval reports are committed.

## 7. Evaluation loop (Dev A)

1. E0 baseline scores on train/val (Dev A, no training).
2. For each finished run: `08_evaluate.py --model <run> --split val` → `reports/eval_<run>_val.md/json` (bootstrap intervals over clips).
3. Dev B picks the winner of each wave from those reports, which decides the next wave.
4. Final model: `07_calibrate.py` (Dev B) → `thresholds.yaml` → scores on test, staged_test and OOPS → `08_evaluate.py` once per split.
5. S10: both laptops run `08_evaluate.py` on the same run and split; the numbers must match exactly (a check that the code, not the machine, decides the result).

## 8. Checklist

- [ ] Dev B: `--threads` flag in `06_train.py` (before Mode 3)
- [ ] Both: Step 0 benchmark filled in; mode chosen
- [ ] Both: `windows_summary.csv` identical
- [ ] Wave 1 (E1–E3) done; val reports written; best of E2/E3 chosen
- [ ] Wave 2 (E4–E5) done; best of E4/E5 chosen
- [ ] Wave 3 (E6–E7) done; E7 outcome written into `reports/occlusion_realism.md`
- [ ] Measured times copied into the master plan's "Training time" table
- [ ] Final model calibrated, frozen and evaluated once on test, staged_test and OOPS
