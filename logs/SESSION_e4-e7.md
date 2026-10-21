# Laptop A session record: E4–E7 (2026-10-17 → 2026-10-18)

Untracked on purpose (handoff rule 0.1.4: never commit `models/`, `logs/`, `data/`, parquet).

## Where things are

- Code and reports: commit `db96d48` on branch `a/e4-e7` (pushed to origin; PR to `main` not yet confirmed).
  All 14 new runs were trained from clean `760f123` on laptop `Khush_Mehta`.
- Runs on disk: `models/e1..e7_*` (27 runs) and `reports/scores/<run>/val.parquet`.
  - E1–E4 (13 runs) are Dev B's (`23288a1-dirty`, `AnanthHN_Laptop`), from `C:\fall-data\E1E4ZIP\devB_e1-e4.zip`.
    `check_runs.py` accepts that one dirty commit via `ACCEPTED_DIRTY`.
- Backups (outside the repo):
  - `C:\fall-data\outgoing\devA_e4-e7.zip`: 16 runs (the 14 new ones, plus e4_gru_s3 and e4_tcn_s1 calibrated here),
    `models/<run>/*` + `reports/scores/<run>/val.parquet`. Unzip at the repo root, the same way as Dev B's zip.
    Upload to the Drive folder `fallsense-data/runs/` so Dev B has it.
  - `C:\fall-data\outgoing\laptopA_e4-e7_logs.zip`: a copy of this `logs/` folder.

## Machine setup that worked (Laptop A)

- Windows power mode: **Best performance**, on AC. OneDrive left running; no lock failures.
- Launcher: `scripts/run_queue.ps1` → `scripts/run_one.ps1` (train → calibrate → eval per run), 60 s gap between starts.
- `maxjobs` 6 (`logs/maxjobs.txt`). It never limited anything: runs take 5–8 min, so at most 5 ran at once.
- Threads / workers:
  - Stage 1 (6 runs): 2 threads / 3 workers. A 1 thread / 4 workers trial was faster (16–18 s/epoch vs 21–31).
    The augmentation workers are the bottleneck, not torch.
  - Stages 2–4 (2–3 runs): **3 threads / 4 workers**, about 86% CPU and no swapping.
  - For next time: 5–6 concurrent runs → 1 thread / 4 workers; 3 or fewer runs → 3 threads / 4 workers.
- Throughput: 1 run alone ≈ 3,840 train windows/s; 5 at once ≈ 12,600/s (3.3×). About 3–4 GB of RAM per run.
- Total training time was about 36 min, roughly 3× faster than the handoff estimate.
- Full history: `logs/tuning.md`. Decisions with numbers: `logs/decisions.md`. Final checks: `logs/check_runs_final.txt`.

## Results (val; primary = false FALL alerts/h at ≥75% FALL sensitivity, lower is better)

| Row | Primary (min–max) | Sens. FALL | OF-Syn | Elderly | OCCU |
| --- | --- | --- | --- | --- | --- |
| E4 GRU | 148 (133–173) | 78.7% | 78.5% | 65.3% | 47.9% |
| E4 TCN | 276 (264–288) | 75.2% | 82.6% | 72.0% | 43.8% |
| E5 GRU | 44 (37–51) | 85.9% | 91.1% | 90.7% | 37.5% |
| E5 TCN | 41 (37–45) | 84.5% | 88.1% | 82.7% | 35.4% |
| E6 GRU | 40 (36–49) | 84.5% | 87.4% | 84.0% | 39.6% |
| E7 GRU | 50 (48–52) | 84.5% | 87.4% | 82.7% | 41.7% |

- **Final model:** E5 GRU (core + occu + of-syn, `--augment`, no `--occlude`). Freeze **e5_gru_s1**:
  37 false alerts/h at 75% sensitivity, 89.6% calibrated sensitivity, 96.7% OF-Syn, 100% elderly.
- E4 → E5: OF-Syn is a clear win (the ranges don't overlap; elderly +25 points).
- GRU vs TCN: tied on false alerts; GRU wins the tie-break on elderly sensitivity.
- E6 (no CAUCAFall): no measurable drop, but val has only 4 CAUCAFall falls.
- E7 (occlusion): dropped. +4 points on OCCU from one seed (2 of 16 falls), no gain at equal sensitivity. This was a judgement call.
- Only val was used. test, staged_test and OOPS were not opened.

## Open items for the next session

1. e5_gru_s1 operating point: calibrated threshold 0.30 → 113 false alerts/h. A threshold of 0.70 gives 37/h at 77% sensitivity.
   The goal is 0.5/h. Choosing the demo threshold is Dev B's call.
2. OCCU fell with OF-Syn (47.9% → 37.5%) and E7 didn't recover it. See the "revisit" notes in `reports/occlusion_realism.md`.
3. `reports/experiments.csv`: the E4 TCN row mixes provenance (s1 is Dev B's, s2 and s3 are Laptop A's), but only the first run's commit and laptop are shown.
   The fix in `experiments_table.py` is Dev B's change.
4. `check_runs.py` compares thresholds by content for Dev B's runs, because unzipping made their eval reports look older than the thresholds.
   Re-evaluating e4_gru_s1 and e4_gru_s2 gave byte-identical output.
5. Open the PR `a/e4-e7` → `main` (merge commit, don't delete the branch) if that isn't done yet.
