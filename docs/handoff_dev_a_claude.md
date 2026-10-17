# Handoff: automated training and evaluation queue (E4 → E7) on Laptop A

You are Claude, running the fallsense fall-detection training queue (repo `KhushM7/Sotonship26`, Windows, PowerShell, Python 3.11 venv in `.venv`) **unattended on Laptop A**. Dev B's Claude ran E1–E4 on Laptop B and stopped part-way through E4. You finish **E4, then E5, E6 and E7**: train, calibrate, evaluate, decide and write up. Keep this laptop's CPU busy the whole time, and run training, calibration and evaluation in parallel.

**This is an automated session with stage checkpoints.** Within a stage, make every decision yourself, using the rules in this document, and keep going. At the end of every stage, report to Dev A and **wait** for a reply before starting the next stage (section 2.1). Otherwise contact Dev A only in the cases listed in section 0.2. Never contact Dev B: anything that needs Dev B goes into the final findings note.

**Not every experiment has to run.** Dev A is short on time and may stop after any stage. The stages are ordered so that the best model is found as early as possible, and every stage ends with a usable "best model so far". A row that wasn't run is reported as "not run (time)", not as a failure.

Read this whole document before you run anything. Read `docs/Training Run Plan (Fall Detection).md` and the "Occlusion handling" section of either dev plan only if a rule here is unclear. Where they disagree with this document, this document wins: it records decisions made after they were written.

## 0. Ground rules

### 0.1 Fixed rules

1. **Do not commit or push.** A hook blocks `git commit` and `git push`. At the very end, give Dev A the exact `git` / `gh` commands (section 7). Merges use **Create a merge commit**, and branches are never deleted.
2. **Use val only.** Never open, score or evaluate `test`, `staged_test` or `oops`.
3. **Every run comes from a clean commit on `main`.** `models/<run>/config.yaml` records `git_commit` (from `git describe --always --dirty`) and `laptop`. A run whose `git_commit` ends in `-dirty` does not count. `--dirty` counts staged and modified tracked files but ignores untracked ones, so:
   - Before the first launch, `git status --porcelain --untracked-files=no` must print nothing. If it prints anything, stop and tell Dev A (section 0.2).
   - While training runs, **do not edit any tracked file**: no plan docs, no `reports/experiments.csv`, no `reports/eval/summary_val.*`, and no `reports/occlusion_realism.md`. The per-run `reports/eval/eval_<run>_val.*` files that `08_evaluate.py` writes are new, untracked files, so they're fine. The ones for `e4_gru_s1` and `e4_gru_s2` already exist and are tracked; never re-evaluate those two while anything is training.
   - Write the summary tables to the scratch folder `logs/summary/` while training runs (section 5). Write the tracked versions only after the last training run has finished.
   - New orchestration scripts are untracked, so they're fine.
   - The exception is Dev B's E1–E4 runs, whose `git_commit` reads `23288a1-dirty`. That is known and accepted.
4. **Never commit anything under `data/`, `models/`, `logs/` or `reports/scores/**/*.parquet`.** `models/e*/` and `logs/` are not in `.gitignore`, so never run `git add -A`, `git add .` or `git add models`.
5. **Add type hints to new code only.** Never add or edit types in existing code.
6. **Don't edit Dev B's files:** `06_train.py`, `07_calibrate.py`, `windows.py`, `dataset.py`, `models.py`, `calibrate.py`, `experiments_table.py` and `configs/default.yaml`. If one of them needs a change, don't stop. Work around it in your own orchestration code, and record the issue in the findings note.
7. **Never change these settings,** so that results stay comparable with E1–E4:
   - `06_train.py` defaults: lr, weight decay, batch 256, label smoothing, class weights, 40 epochs, patience 8
   - `07_calibrate.py` defaults: `--target-fall 0.90 --target-any 0.95 --plateau 0.03`, `--boot 1000`
   - `08_evaluate.py` `--boot 1000`
   - `eval_summary.py --target 0.75`

   Speed comes only from parallelism, thread counts and scheduling.

### 0.2 When to contact Dev A

Stop the queue and message Dev A only when one of these happens:

1. **The tree is dirty before the first launch** (rule 0.1.3), or **`pytest` fails** on `main`.
2. **The windows don't match after the rebuild** (section 1, step 3): `git diff --exit-code reports/windows_summary.csv` shows a difference, or val isn't 12,454 windows.
3. **A run fails twice with the same seed.** Before you count it as a failure, rule out causes on your side: out of memory, OneDrive file locks, or a launcher bug. Fix and rerun those without asking.
4. **A completion check fails** (section 4.1) and you can't fix it by rerunning a step.
5. **A stage ends** (section 2.1): report and wait.
6. **The end of the session:** give the findings note and the commit commands.

Everything else, decide yourself and write the decision down: concurrency, the order within a stage, the E4/E5 pick, tie-breaks, the E6/E7 config, a missing zip (section 1, step 5).

## 1. Start-up (no input needed; about 10 min)

The PRs `b/train-fixes` and `b/calibrate` (#26) are already on `main` as of `02b5e36`. They include `07_calibrate.py`, `experiments_table.py` and Dev B's re-evaluated `reports/eval/*` for E1–E4. On this laptop, `pytest` gave **97 passed, 1 skipped** on that commit.

Run these in order. Steps 4 and 5 can run while step 3 builds.

1. **Pull and test:**

   ```powershell
   git checkout main; git pull --ff-only; .venv\Scripts\python -m pytest -q
   ```

   Expect at least 97 passed and 0 failed. Then run `git status --porcelain --untracked-files=no`, which must print nothing.
2. **Check the machine** (section 3.1): mains power and OneDrive. Fix what you can and note the rest. Neither one blocks the queue.
3. **Rebuild the windows.** `05_build_windows.py` and `windows.py` changed in #26, so the files in `data/windows` from before the pull are stale.

   ```powershell
   .venv\Scripts\python scripts\05_build_windows.py
   ```

   Then `git diff --exit-code reports\windows_summary.csv` must show no difference. Val must have **12,454** windows (train 53,429, test 14,673, oops 38,872). If anything differs, go to section 0.2, item 2.
4. **Check the baseline:** `reports\scores\baseline\val.parquet` must exist. If it doesn't, run `scripts\baseline_e0.py`. Its output is identical on every laptop.
5. **Dev B's runs: already unzipped, so this is plan A** (2026-10-17, from `C:\fall-data\E1E4ZIP\devB_e1-e4.zip`, after a path-safety check). `models\e1_gru_s1 … e4_tcn_s1` and `reports\scores\e1…e4_*\val.parquet` are in place. `e4_gru_s3` and `e4_tcn_s1` have no `thresholds.yaml` yet, as expected. Just check that these 13 model folders exist, and use plan A. The rest of this step applies only if they've gone missing.

   **Find Dev B's runs:** `devB_e1-e4.zip`, from the shared Drive folder `fallsense-data/runs/`.
   - Look for it automatically, without asking, in `C:\fall-data\E1E4ZIP\`, the repo root, `data\incoming\` and `$env:USERPROFILE\Downloads\`. Check that it holds only `models/e*_*_s*/…` and `reports/scores/e*_*_s*/…`, with no absolute paths and no `..`.
   - Unzip it at the repo root. It creates `models\e1_gru_s1 … e4_tcn_s1` and `reports\scores\e*_*_s*\val.parquet`, and overwrites the stale E1–E3 parquets on this laptop.
   - **If the zip is found (plan A):** the stages are as in section 2.
   - **If it isn't found (plan B):** don't wait, and don't retrain E4 up front.
     - In stage 1, compare E5 with the E4 GRU numbers in the committed `reports/eval/summary_val.md` (2 seeds; section 8) instead of with local E4 runs. Say in the stage report that the comparison is with Dev B's committed numbers.
     - Retraining E4 here (`e4_gru_s1..3`, `e4_tcn_s1..3`, from the clean commit) moves to the optional stage 4.
     - Leave E1–E3 out of every summary `--runs`, because the local E1–E3 parquets don't match Dev B's re-evaluated JSONs. In the write-up, quote E1–E3 from the committed `reports/eval/summary_val.md`, and keep their rows in `reports/experiments.csv` (section 6, step 2).
   - Send Dev A one non-blocking line saying which plan you are on. If the zip turns up later, switch to plan A.

## 2. The queue

"Core" = `GMDCSA24 caucafall le2i mcfd`. `occu` and `of-syn` are added on top.

| Row | Model | Train sets | Flags | State at handoff | Your job |
| --- | --- | --- | --- | --- | --- |
| E1 | GRU | core | `--no-augment` | 3/3 trained and evaluated | none |
| E2 | GRU | core | `--augment` | 3/3 | none |
| E3 | TCN | core | `--augment` | 3/3 | none |
| E4 | GRU | core + occu | `--augment` | 3/3 trained, s3 not calibrated | calibrate and evaluate s3 |
| E4 | TCN | core + occu | `--augment` | s1 trained but not calibrated | calibrate and evaluate s1; train, calibrate and evaluate s2 and s3 |
| E5 | GRU | core + occu + of-syn | `--augment` | – | s1–s3 |
| E5 | TCN | core + occu + of-syn | `--augment` | – | s1–s3 |
| E6 | best of E4/E5 | that config minus `caucafall` | same as best | – | s1–s3, named `e6_<model>_s<seed>` |
| E7 | best of E4/E5 | same as best | same plus `--occlude` | – | s1–s3, named `e7_<model>_s<seed>` |

### 2.1 Stages, ordered so the best model comes first

Each stage is chosen by how likely it is to give a **better final model**, not by its E-number. Rows that only measure something without improving the model (E6, completing the E4 TCN ablation) come last.

| Stage | What runs | Training runs | Why this order | Est. time |
| --- | --- | --- | --- | --- |
| **1. E5** | Calibrate and evaluate `e4_gru_s3` and `e4_tcn_s1` (no training, no slot). Train `e5_gru_s1..3` and `e5_tcn_s1..3`, alternating GRU and TCN | 6 | OF-Syn targets the biggest known weakness: missed OF-Syn and elderly falls, and false alerts on deliberate low movements. E5 is the most likely row to produce the best model. Both models run, because GRU and TCN were tied | 30–40 min |
| **2. E7** | `e7_<best>_s1..3`: the stage 1 winner (section 4.2, rule 4) plus `--occlude` | 3 | The only remaining change that can still improve the model: robustness to occlusion, which the live camera needs | 25–35 min |
| **3. E6** | `e6_<best>_s1..3`: the best config so far (the E7 config if E7 was kept, otherwise the stage 1 winner) minus `caucafall` | 3 | Doesn't improve the model. It measures how much worse the model is in an unseen home, for the write-up | 15–25 min |
| **4. Complete E4 (optional)** | Train `e4_tcn_s2` and `e4_tcn_s3`. Under plan B, train all six E4 runs here instead | 2 (plan B: 6) | Ablation completeness only: it makes the E4 → E5 comparison rest on 3 seeds per model | 10–20 min |

**Within a stage, run everything in parallel.** Queue all of a stage's runs at once, put the longest first, and start with `maxjobs` 6 (section 3.3), so a stage normally runs as one wave.

**At the end of every stage, report and wait:**

1. Make sure the queue is empty, and that no `06_train`, `07_calibrate` or `08_evaluate` process is still running. Calibration and evaluation for the stage must have finished.
2. Run `check_runs.py` on the stage's runs, and update the scratch summary (section 5).
3. Apply the decision rules (section 4.2) to everything that exists so far, and write the result to `logs\decisions.md`.
4. If the PushNotification tool is available, send one notification: `Stage <n> (<E-row>) done – reply to continue`.
5. End your turn with a **stage report** (at most about 20 lines):
   - what ran, the wall time, and anything that failed or was retried
   - a compact table of the rows so far: the primary number with its min–max range, Sens. FALL, OF-Syn, Elderly and OCCU sensitivity
   - the decision made at this stage, with its numbers
   - **the best model so far:** its config and the seed you'd freeze, with a one-line reason
   - what the next stage would run, its estimated time, and what it could change
   - the choices: **"continue"** (run the next stage), **"skip to stage N"**, or **"stop"** (go straight to the write-up with what exists)
6. Start nothing more until Dev A replies. When the reply comes, carry on from the current state: the scripts are idempotent, and finished runs are never redone.

If Dev A says "stop", do the write-up (section 6) on the completed rows. Report every row that wasn't run as "not run (time)", and still give the commit commands.

## 3. Laptop A and the parallel setup

### 3.1 Hardware (measured on 2026-10-17)

| Part | Value | What it means here |
| --- | --- | --- |
| Machine | HP ZBook 8 G1i 16" mobile workstation | Laptop cooling: it slows down under long full load, so tune on total throughput, not per-run speed |
| CPU | Intel Core Ultra 9 285H: 16 cores / 16 threads (6 P-cores + 8 E-cores + 2 low-power E-cores), no hyper-threading | About 11–12 P-core equivalents. Augmentation workers (single-threaded Python) run well on E-cores, and torch's math runs best on P-cores. Don't oversubscribe heavily |
| RAM | 64 GB DDR5-5600 (2 × 32 GB), about 35 GB free at idle | RAM is not the limit. One run (main process plus 3 workers) uses about 2–3 GB. Keep at least 8 GB free |
| GPU | NVIDIA RTX 500 Ada (4 GB), Intel Arc 140T | **Not used.** The venv has `torch 2.14.1+cpu`, the models are tiny, augmentation on the CPU is the bottleneck, and E1–E4 ran on CPU. Don't install CUDA torch |
| Disk | SK hynix PC801 1 TB NVMe, about 295 GB free | Not a bottleneck |
| Power | "HP Optimized (Modern Standby)" plan | Must be on mains power, with the Windows power mode set to **Best performance** |
| Repo location | `C:\Users\khush\OneDrive\...` | OneDrive syncs every write: `train_log.csv` is rewritten every epoch, and logs are written constantly. That costs CPU, and a sync lock can make a write fail |

**Machine checks for step 2 of section 1** (fix what you can and note the rest; neither blocks the queue):

- **Mains power:** `(Get-CimInstance Win32_Battery).BatteryStatus` should be `2`, meaning AC. If it isn't, tell Dev A in a non-blocking line, and keep going.
- **OneDrive:** `Get-Process OneDrive -ErrorAction SilentlyContinue`. If it's running, Dev A should already have paused syncing (start prompt). If a write fails because a file is locked, retry once after 5 s. The launcher handles this.

### 3.2 Process settings (every Python process the launcher starts)

- Environment for training: `OMP_NUM_THREADS=<threads>`, `MKL_NUM_THREADS=<threads>`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `PYTHONUNBUFFERED=1`.
  - Without these, each run's numpy start-up and each DataLoader worker can spawn 16 BLAS threads, which thrashes the CPU when 5 or more runs share it.
  - DataLoader workers already set torch to 1 thread themselves.
- Environment for calibration and evaluation: all four thread variables set to `1`. Both steps are mostly single-threaded bootstrap work.
- Leave process priority at Normal. Don't use `-WindowStyle Hidden` with low priority: on a hybrid CPU, Windows can move low-priority background processes onto the low-power E-cores.

### 3.3 Starting configuration

| Setting | Start value | Why |
| --- | --- | --- |
| Training runs at once (`maxjobs`) | **6** | Stage 1 has 6 runs and stages 2–3 have 3, so every stage runs as one wave. 6 × (2 + 3) = 30 tasks on 16 cores. The workers spend much of their time waiting on Python and inter-process transfer, so this keeps every core busy. If it thrashes, section 3.4 lowers it |
| `--threads` | **2** | A GRU or TCN at batch 256 gains little beyond 2 torch threads, and 2 leaves P-cores for other runs |
| `--workers` | **3** | Per-window augmentation is the bottleneck for E2–E7 |
| Gap between training starts | **60 s** | Start-up (features for every train and val window) runs single-threaded in one process: about 1 min for core, several minutes for E5. Staggering the starts overlaps start-up with other runs' training |
| Calibrate and evaluate | Start immediately when a run's training ends, **outside** the training slots | 1–4 min each, mostly one core. Keeps the slot free for the next training run |

### 3.4 Tune as you go (automatic, no need to ask)

The launcher reads `logs\maxjobs.txt` before every start, so you can change concurrency without restarting it. Changes to `--threads` and `--workers` apply to new launches only. Record each change and its reason in `logs\tuning.md`.

- **First measurement.** After the first stage 1 runs have each logged epoch 3 (epoch 1 includes worker start-up, so ignore it), take 30 s of CPU and RAM samples with `Get-Counter '\Processor(_Total)\% Processor Time','\Memory\Available MBytes' -SampleInterval 5 -MaxSamples 6`. Then compute **total throughput** = Σ over the running training runs of (train windows ÷ `seconds` of their latest epoch). Get the train window counts from the `train N windows` line in each training log.
- **Adjust,** one step at a time, measuring again about 3 epochs after each change:
  - Total CPU under 85% and more than 8 GB of RAM free → raise `maxjobs` by 1.
  - The last increase raised total throughput by less than 8% → lower `maxjobs` back by 1 and stop increasing. More runs at once no longer help.
  - CPU at 97% or more and total throughput falling → lower `maxjobs` by 1.
  - Less than 8 GB of RAM free → lower `maxjobs` by 1. That is unlikely here.
- **Workers vs threads:** for one new launch, try `--threads 1 --workers 4`. If its s/epoch is lower than its neighbours' at the same data size, use it for the remaining launches.
- **E7 / `--occlude`:** each worker also holds a `ClipBank` copy (about 50 MB, under 1 GB per run). That's still fine for RAM, but each epoch is slower. Measure stage 2 again once its 3 runs reach epoch 3. With only 3 runs, use `--threads 3 --workers 4` for stages 2–4, so they still fill the cores.
- Fill in row A of the Step 0 table in the Training Run Plan at the end (it's a tracked file): s/epoch at core + occu and at core + occu + of-syn under the chosen concurrency, s/epoch with `--occlude`, peak RAM, and swapping (none expected).

**Expected timing.** Laptop B, with 8 cores and solo runs, took 12–19 s/epoch on core or core + OCCU, and best epochs averaged 7–9, so runs early-stop at about epoch 15–17.

| Run type | Approx. s/epoch here | Approx. time per run, including start-up |
| --- | --- | --- |
| E4 | 20–30 s | 8–12 min |
| E5 | 45–70 s | 15–25 min |
| E7 | about 1.3–1.8× its base row | longest |

Stage 1 should take about 30–40 min, stage 2 about 25–35 min, stage 3 about 15–25 min and stage 4 about 10–20 min. That's 1.5–2 h for everything, plus the time spent waiting at checkpoints.

### 3.5 Orchestration scripts (write these yourself; they are untracked until section 7)

**`scripts\run_one.ps1 -Run <name> -TrainArgs <string> [-Threads 2] [-Workers 3]`** runs the full pipeline for one run. It is idempotent, so you can always run it again to resume:

1. **Train:** skip this step if `models\<run>\val_logits.npy` exists.
   - If `models\<run>` exists without `val_logits.npy`, it is a half-finished run. Move it to `models\_partial\<run>_<timestamp>` first.
   - Write `train` to `logs\<run>.stage`, set the environment from section 3.2, and run `06_train.py <TrainArgs> --threads N --workers M --out models\<run>`. Send the output to `logs\<run>.train.log`.
2. **Calibrate:** skip this step if `models\<run>\thresholds.yaml` is newer than `val_logits.npy`.
   - Write `calib` to the stage file and run `07_calibrate.py --run <run>`. Send the output to `logs\<run>.calib.log`.
3. **Evaluate:** skip this step if `reports\eval\eval_<run>_val.json` is newer than `thresholds.yaml`.
   - Write `eval` to the stage file and run `08_evaluate.py --model <run> --split val`. Send the output to `logs\<run>.eval.log`.
4. On success, write `logs\<run>.done`. If any step exits non-zero, write `logs\<run>.failed` containing the step name and the exit code, and stop.

**`scripts\run_queue.ps1 -Queue logs\queue.txt`** keeps the training slots full. Each line of the queue file is `<run>|<train args>`. Write one stage's lines at a time. The runner exits when every line is `.done` or `.failed` and nothing is running, which is your signal that the stage has ended. Every 15 s it:

1. **Counts training jobs:** live job processes whose stage file says `train`. Calibration and evaluation don't count.
2. **Starts the next job** with `Start-Process pwsh -ArgumentList '-NoProfile','-File','scripts\run_one.ps1',...` if all of these hold:
   - training jobs < `logs\maxjobs.txt`
   - at least 60 s have passed since the last training start
   - the next queue line is not `.done`, not running, and not `.failed` twice

   Calibrate-only entries (for example `e4_gru_s3|`, with empty train args, whose `val_logits.npy` already exists) start immediately, without waiting for a slot or the gap.
3. **Retries a failure once,** with the same seed, if the log shows a cause on your side: `MemoryError`, `PermissionError` or a locked file. Otherwise it marks the run `.failed` and moves on.
4. **Never starts two jobs on the same run.** Track each running run name together with its PID.
5. **Writes one status line** to `logs\queue_status.txt`, for example `running: e5_gru_s1(train ep 6) … | done: … | failed: …`.

**`scripts\check_runs.py <run> ...`** is the completion check in section 4.1, as code. It prints one PASS or FAIL line per item for each run, and exits non-zero if any item fails. Use type hints, since this is new code.

**How you supervise:**

- Start `run_queue.ps1` with `run_in_background`.
- Use the Monitor tool, or a background until-loop, to wake when a new `.done` or `.failed` file appears, or every 10–15 min as a heartbeat. Don't poll more often than that.
- Each time you wake:
  1. Run `check_runs.py` on the newly finished runs.
  2. Update the scratch summary (section 5).
  3. Apply the tuning rules in section 3.4.
  4. Apply the decision rules in section 4.2 as soon as rows complete.

**Stopping:**

1. Stop `run_queue.ps1` first, so it can't start anything new.
2. Then stop the Python processes. Find them with `Get-CimInstance Win32_Process -Filter "Name='python.exe'"` and match `06_train|07_calibrate|08_evaluate` in the command line. On Windows, child processes can outlive the shell that started them.
3. Move any half-finished `models\<run>` folder to `models\_partial\`.

## 4. Completion and decisions

### 4.1 When a run is complete (check with `check_runs.py`, not by eye)

- `models/<run>/`:
  - `model.pt`, `feat_stats.npz` and `confusion.png`
  - `config.yaml`: `git_commit` does not end in `-dirty` (except Dev B's `23288a1-dirty`), and `laptop` is set
  - `train_log.csv`: has the `val_all_*` columns
  - `val_logits.npy`: exactly 12,454 rows
  - `thresholds.yaml`: has a `calibration:` block
  - `calibration.csv`
- `reports/scores/<run>/val.parquet`:
  - 12,454 rows
  - passes `fallsense.scores.validate_scores`
  - modified after `val_logits.npy`
- `reports/eval/eval_<run>_val.{md,json}` and `_coverage.png`: modified after `thresholds.yaml`.
- **Evidence to collect for each run** (quote it in the findings note, in a compact table):
  - the `best val AP(falling) …` line from training
  - the `temperature … ECE …` and `val: FALL sensitivity …` lines from calibration
  - the `(CHECK band)` or `(one threshold)` line from evaluation

**An experiment row is complete** when all 3 seeds are complete. If a seed fails, rerun it with the same seed. Never substitute a different seed.

### 4.2 How to compare rows and decide (apply this yourself as soon as the data exists)

1. **The primary number** is **"Fewest false / h at ≥ 75% sens."** in the summary: the mean over seeds, with the min–max range. Lower is better.
   - Fixed thresholds put seeds at very different points: 59–117 false alerts/h within one experiment at 0.5/0.5. That's why rows are compared at a sensitivity every run reaches, not at a fixed threshold.
2. **Secondary numbers:**
   - FALL sensitivity at the calibrated threshold
   - OF-Syn, Elderly and OCCU sensitivity
   - core sensitivity and core false alerts/h
   - all-val window AP
3. **Keep an addition** (OCCU in E4, OF-Syn in E5, occlusion in E7) only if one of these holds:
   - **It improves the primary number by more than the seed spread:** the new mean is below the old row's min–max range, or the two ranges don't overlap.
   - **It clearly improves elderly, OF-Syn or OCCU sensitivity,** by more than 10 points on the mean, without raising the primary number by more than the spread.

   Otherwise, record "no clear gain" and keep the simpler config.
4. **Best of E4/E5:**
   1. Apply rule 3 to E5 vs E4, for GRU and TCN separately.
   2. Compare GRU vs TCN on the winning data the same way.
   3. If they're tied within the spread, prefer the higher elderly sensitivity (the target population), and say that this was a tie-break.
   4. Write the decision, with its numbers, to `logs\decisions.md` the moment you make it.
   5. **With partial data** (plan B, or E4 TCN with 1 seed, before stage 4), decide on what exists. A single-seed row has no spread, so treat its value as a point inside a range equal to the other model's spread. Say that the decision is provisional, and revisit it if stage 4 runs.
5. **E6** ("how much worse in an unseen home"):
   - Compare the `caucafall` rows of the per-dataset breakdown in `eval_e6_*_val.md` with the same rows for the chosen E4/E5 row.
   - Note that CAUCAFall val is small (8 clips, 4 falls).
6. **E7** vs the chosen E4/E5 row:
   - Compare on all of val (primary number), on OCCU val sensitivity, and on false alerts/h outside OCCU (per-dataset rows).
   - Keep occlusion only if OCCU or hidden-fall sensitivity rises by more than the spread, and false alerts elsewhere don't rise by more than the spread.
7. **Done means:**
   - Every row in the stages that ran is complete under 4.1. Rows in skipped stages are listed as "not run (time)".
   - Every decision is in `logs\decisions.md` with its numbers.
   - `check_runs.py` passes for every reported run.
   - No reported run is `-dirty` (except Dev B's), uncalibrated, or missing from the summary.

## 5. Scratch summaries during training (tracked files stay untouched)

`eval_summary.py` and `experiments_table.py` write to tracked files, so don't run them while training is going on (rule 0.1.3). Instead, while training runs, get your numbers like this:

- **Experiment table:** after each row completes, copy `scripts\experiments_table.py` and `scripts\eval_summary.py` to `logs\tools\` and change only the output path, to `logs\summary\`. Run the copies.
- **Eval summary:** run the copy with `--target 0.75 --runs baseline <every complete run>`, in a separate process, so it doesn't block the supervisor. Always pass `--runs`. Without it, the script reads every report in `reports\eval`, including stale ones.
- Decide from `logs\summary\summary_val.md`.

## 6. Write-up (after the last stage, or as soon as Dev A says "stop"; never while a training process is running)

Do this once, on whatever has completed. Include only complete rows. List the rows from skipped stages as "not run (time)" in the findings note and in the `occlusion_realism.md` E7 section. If E7 didn't run, write "E7 not run (time)" there.

1. **The real summary:**

   ```powershell
   .venv\Scripts\python scripts\eval_summary.py --target 0.75 --runs baseline <all complete runs, E1–E7>
   ```

   Under plan B, leave out E1–E3, and say so in the file's text and in the findings note.
2. **The experiments table:** `.venv\Scripts\python scripts\experiments_table.py`.
   - Under plan B, this drops the E1–E3 rows, because their model folders aren't on this laptop. Put them back from `git show HEAD:reports/experiments.csv`, keeping only the `e1_*`, `e2_*` and `e3_*` rows and the same columns, with a small script of your own.
3. **Re-evaluate `e4_gru_s1` and `e4_gru_s2`** only if their tracked reports are older than their `thresholds.yaml`. Normally they aren't.
4. **`reports/occlusion_realism.md`**, "E7 outcome" section (the plans require it):
   - E7 vs its base on all of val, on OCCU val, and false alerts/h elsewhere
   - all run names
   - the keep/drop decision with its reason
   - if E7 doesn't help, say so and point to the existing "what to revisit" notes
5. **Plan docs:**
   - Tick the Training Run Plan section 8 items that are now true, plus the matching E-row items in both dev plans, in the existing ✅ / ⏳ style with a short note.
   - Fill in row A of the Step 0 table.
   - Copy the measured times into the master plan's "Training time" table.
6. **Final checks:** run `check_runs.py` on every reported run, and paste its output. Run `.venv\Scripts\python -m pytest -q` if you added Python code.
7. **The findings note,** in chat, for Dev A and Dev B:
   - which plan you used (A or B)
   - the concurrency you settled on, and the measured throughput
   - the E4 → E5 decision with numbers, GRU vs TCN, E6's drop, and E7's verdict
   - the chosen final config, and which seed's run you'd freeze, with the reason
   - anything that broke, looks wrong, or needs Dev B: in particular, whether E5 reduced false alerts. If it didn't, the earlier suggestion was more class weight on controlled_low near-misses, which is Dev B's change to make.

## 7. Commit commands for Dev A

Use the branch `a/e4-e7`. Stage only these:

- `reports/eval/*`
- `reports/experiments.csv`
- `reports/occlusion_realism.md`
- the edited plan docs
- `scripts/run_one.ps1`, `scripts/run_queue.ps1` and `scripts/check_runs.py`

Never stage `models/`, `logs/`, `data/` or parquet files. Print the exact commands as one block, then stop:

```powershell
git checkout -b a/e4-e7
git add <files>
git status
git commit -m ...
git push -u origin a/e4-e7
gh pr create ...
```

## 8. What Dev B found on E1–E4 (what to expect)

Mean over seeds. "False/h @75%" is the fewest false FALL alerts per hour among thresholds that keep FALL sensitivity at 75% or more.

| Row | Window AP (all val) | Sens. FALL (calibrated) | False/h @75% | Core sens. | OF-Syn sens. | Elderly sens. | OCCU sens. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 rule | 0.10 | 68.8% | – | 68.4% | 81.1% | 80.0% | 0.0% |
| E1 GRU no aug | 0.61 | 75.5% | 175 (133–196) | 93.0% | 74.4% | 61.3% | 39.6% |
| E2 GRU aug | 0.64 | 77.1% | **131 (113–168)** | 93.0% | 75.6% | 64.0% | 47.9% |
| E3 TCN aug | 0.64 | 74.8% | 282 (152–454) | 86.0% | 76.3% | 58.7% | 39.6% |
| E4 GRU aug + OCCU (2 seeds) | **0.68** | **79.5%** | 156 (139–173) | 92.1% | **80.0%** | **68.0%** | 46.9% |

- **No run reaches 90% FALL sensitivity on val.** Sensitivity plateaus at about 76–80%, because core-trained models miss most OF-Syn and OCCU falls. E5 adds OF-Syn and should move this the most.
- **False alerts are the main weakness:** 110–200/h at the calibrated thresholds. They come mostly from deliberate low movements, which is a falling-vs-controlled_low confusion:
  - OF-Syn `lie_down`, and `other` (kneeling and crawling, labelled upright)
  - GMDCSA24 daily activities
- **The 30 s refractory hides about 11 of 144 val falls.** A false alarm more than 3 s earlier in the same clip suppresses the real FALL. That's the event logic working as specified. Note it, but don't change it.
- **The E0 rule still catches more OF-Syn and elderly falls** (81% and 80%), because those falls are often slow. Watch whether E5 closes this gap.
- **The models are overconfident:** T = 1.3–1.8, and ECE drops from 0.09–0.21 to 0.02–0.06 after scaling.
