# Dev B Plan – Model Out (Fall Detection)

Oct 17, 2026 · @K

## How to use this doc

Dev B owns **model out**: config and environment check, features, windows, training, calibration, the event logic and the live demo. Dev A follows a matching doc ([Dev A Plan – Data In](https://claude.ai/code/artifact/9537df20-f95f-4ef6-b39f-a35d675f5684)) and creates the repo; the full reference is the [master build plan](https://claude.ai/code/artifact/a8b9bcc7-5227-445b-ba58-900b7e0cc8bd).

Work top to bottom and tick each box. Wherever a box marked **SYNC** appears, stop: say out loud what you are handing over or waiting for, the other dev runs the listed check and confirms out loud, and only then do you both tick it. The IDs S0–S15 are identical in both docs.

| Sync | Direction | What is handed over | How the receiver confirms it |
| --- | --- | --- | --- |
| S0 | Both | All overnight downloads verified | Each reads out `omnifall verify` results for their own datasets |
| S1 | A → B | Repo created, B added as collaborator, skeleton pushed | B clones and pushes a test branch |
| S2 | A ↔ B | A: `requirements.txt`, `requirements.lock`. B: `config.py`, `configs/default.yaml`, `00_check_env.py` | Both run `00_check_env.py`; every line OK; both read out the mediapipe and torch versions |
| S3 | Both | Data contracts 1–4 agreed, `schema_version: 1` | Read the contracts section together and both say "agreed" |
| S4 | A → B | `io_video.py`, `pose.py`, `02_inventory.py`, `03_extract_skeletons.py`; pose model choice | B runs inventory and a 5-clip smoke extraction on GMDCSA24 and reads out shapes and detection rate |
| S5 | A → B | `labels.py` and its tests | B runs `pytest tests/test_labels.py`; both look at frame labels for one fall clip |
| S6 | B → A | B's skeletons (GMDCSA24, CAUCAFall, MCFD) in the shared folder, plus B's `extract_log.csv` | A unzips; .npz count per dataset matches B's log |
| S7 | A → B | Core QC done: `index.csv`, `qc.csv`, QC plots, Le2i skeletons | B unzips; reads out index rows and `qc_pass` counts per dataset |
| S8 | B → A | `features.py` (`make_features`) and its tests | A runs `pytest tests/test_features.py`; `baseline.py` imports it |
| S9 | B → A | Event logic in `calibrate.py` (`alerts_from_scores`) and the contract-4 score writer | A's `tests/test_metrics.py` passes against it |
| S10 | A → B | `08_evaluate.py`, `metrics.py`, rule baseline (E0) scored on val | B runs the evaluator on E1 val scores; both read out sensitivity and false alerts per hour |
| S11 | A → B | Extras QC'd and shared: OCCU, OF-Syn subset, OOPS; `index.csv` updated | B rebuilds windows and reads out window counts per dataset |
| S12 | Both | Staged recording session done (consent forms, safety, actors) | Both tick off the clip coverage table together |
| S13 | A → B | Staged clips labelled (B double-checks fall times), extracted, `staged_test` rows in `index.csv` | B builds `staged_test.npz` and reads out clip and window counts |
| S14 | B → A | Frozen model: `model.pt`, `feat_stats.npz`, `thresholds.yaml`, score files for test, staged\_test and OOPS | A runs the final evaluation once; both review the report |
| S15 | Both | Live demo at 10+ fps, skeleton-only display; definition-of-done checklist | Walk the checklist together |
| S16 | A → B | `occlusion.py` merged (context, disappearance finder, simulation) and `reports/occlusion_realism.md` | B runs `pytest tests/test_occlusion.py` and reads the realism report |
| S17 | A → B | `tracker.py` merged | B wires `DisappearanceTracker` and `LastSeenState` into `09_live.py`; both see a CHECK when someone ducks behind a chair and stays down |

Compute times in this doc are estimates; replace them with measured numbers as you go.

## Step 0 – Morning download checks

Your laptop holds GMDCSA24, CAUCAFall and MCFD in `C:\fall-data\cache\videos\<dataset>\video\` (30 GB cap).

- [ ] In a new PowerShell window:
  - `$env:OMNIFALL_CACHE_DIR = "C:\fall-data\cache"`
  - `$env:OMNIFALL_DOWNLOAD_DIR = "C:\fall-data\downloads"`
  - `Remove-Item Env:OMNIFALL_ROOT -ErrorAction SilentlyContinue`
- [ ] `omnifall verify GMDCSA24`, `omnifall verify caucafall`, `omnifall verify mcfd` – each reports no missing files.
- [ ] Video counts: `(Get-ChildItem C:\fall-data\cache\videos\<dataset>\video -Recurse -Filter *.mp4).Count` – expect GMDCSA24 160, CAUCAFall 100; note MCFD's count.
- [ ] Free space: `Remove-Item C:\fall-data\downloads\* -Recurse -Force` and delete `C:\Users\anant\Downloads\Dataset CAUCAFall.zip` if still there. Check usage: `"{0:N1} GB" -f ((Get-ChildItem C:\fall-data -Recurse -File | Measure-Object Length -Sum).Sum / 1GB)` is well under 30.
- [ ] Any failure: rerun `omnifall prepare <dataset> --workers 4` (CAUCAFall needs `--archive` with the Mendeley zip).
- [ ] Note each dataset's licence (`omnifall sources <dataset>`) and BibTeX (`omnifall cite <dataset>`) and send them to Dev A for `DATA_CARD.md`.
- [ ] **SYNC S0 (both)** – read your three verify results out loud; Dev A reads theirs (Le2i, OCCU, OF-Syn, OOPS). Both tick only when all seven datasets verify.

## Step 1 – Clone the repo, environment, config and env check

Dev A creates the repo and lockfile; you add the config and the environment check that both laptops must pass.

### 1a. Clone

- [ ] **SYNC S1 (A → B)** – Dev A gives you the repo URL and adds you as collaborator. Run `git clone <repo-url> C:\code\fallsense`, create branch `b/hello`, push it, and tell Dev A. Tick when Dev A sees it on GitHub.

### 1b. Environment (after Dev A pushes `requirements.lock`)

- [ ] From `C:\code\fallsense`:
  - `uv venv --python 3.11 .venv`
  - `.venv\Scripts\activate`
  - `uv pip sync requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu`
  - `uv pip install -e .`
- [ ] Copy `.env.ps1.example` to `.env.ps1`.
- [ ] Link data into the repo: `New-Item -ItemType Junction -Path data -Target C:\fall-data`.
- [ ] Save environment variables once, then close and reopen all terminals:
  - `setx FALL_DATA "C:\fall-data"`
  - `setx OMNIFALL_CACHE_DIR "C:\fall-data\cache"`
  - `setx OMNIFALL_DOWNLOAD_DIR "C:\fall-data\downloads"`
  - `setx OMNIFALL_ROOT "C:\fall-data\cache\videos"`
- [ ] Pose models (use `curl.exe`):
  - `curl.exe -L -o models\pose\pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task`
  - same again with `full` replaced by `lite` in both places.

### 1c. `configs/default.yaml` and `src/fallsense/config.py`

```yaml
schema_version: 1
fps: 15
window: 30                 # frames = 2.0 s
stride_train: 8
stride_train_near_fall: 2
stride_eval: 3
pose_model: models/pose/pose_landmarker_full.task
pose: {num_poses: 1, min_det: 0.5, min_presence: 0.5, min_track: 0.5}
le2i_upscale: false        # Dev A sets this after the benchmark
keypoints: [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
classes: [upright, falling, fallen, controlled_low]
datasets: [GMDCSA24, caucafall, le2i, mcfd, occu, of-syn]
test_only_datasets: [OOPS]
split_config_suffix: "-cs"
seed: 1337
on_floor_seconds: 20
```

- [ ] `config.py`: `load_config(path="configs/default.yaml") -> dict`, plus `DATA = Path(os.environ["FALL_DATA"])` and helpers `skeleton_path(dataset, path)` and `windows_path(split)` that follow the contracts.

### 1d. `scripts/00_check_env.py`

Checks in order, printing OK or FAIL per line:

- [ ] Python is 3.11.
- [ ] `import mediapipe, cv2, torch, omnifall, fallsense` succeed; print mediapipe and torch versions.
- [ ] `shutil.which("ffmpeg")` is not None.
- [ ] `FALL_DATA`, `OMNIFALL_CACHE_DIR`, `OMNIFALL_DOWNLOAD_DIR`, `OMNIFALL_ROOT` are set and `OMNIFALL_ROOT` exists.
- [ ] At least 10 GB free on the drive (`shutil.disk_usage`).
- [ ] Both .task files load in a `PoseLandmarker`.
- [ ] One webcam frame from `cv2.VideoCapture(0)` returns a pose with a person in view.
- [ ] Print `torch.get_num_threads()`.
- [ ] Set up `pytest` (a `tests/conftest.py` that adds `src` to the path if needed) and merge 1c–1d by pull request from branch b/config-env (M3); Dev A reviews, and you both confirm main is green before S2.
- [ ] **SYNC S2 (A ↔ B)** – both run `python scripts\00_check_env.py`; every line OK on both laptops; mediapipe and torch versions match. Tick only when both pass.

### Git habits (both)

Branch per task (`b/features`, `b/windows`), small PRs, the other dev reviews before merge, `.githooks/pre-push` runs `ruff format --check` and `ruff check` on every push (a few seconds; turn it on once per clone with `git config core.hooksPath .githooks`); run `pytest -q` yourself before opening a PR. Type hints are welcome but not enforced, nothing under `data/` ever committed.

### Branches and merges to `main` (identical in both docs)

Each dev works on their own branches and merges small pieces to `main` as soon as another piece needs them. Merge points are numbered **M1–M17** and appear again at the step where they happen.

Rules:

- `main` always runs: the only direct push is M1; every other change goes in through a pull request.
- The other dev reviews and approves every PR; the author's push passed the ruff pre-push check and they ran `pytest -q` first.
- Use squash and merge, then delete the branch.
- Right after any merge, both devs run `git checkout main` and `git pull`, then `git rebase main` on whatever branch they are on.
- Merge small and early; a branch should hold one piece, not a day's work.
- Only the file's owner edits it (see the repo layout). A dev who needs a change in the other's file asks for it or opens a small PR that the owner approves; `configs/default.yaml` belongs to Dev B, so Dev A's `le2i_upscale` change goes through you.
- Never commit anything under `data/`; skeletons, windows and models travel through the shared folder.
- Verbal check for every merge: the author says "M-number merged"; the other dev pulls `main`, runs `pytest -q`, and says "main is green".

| Merge | Who | Branch | What goes in | Merge when | Needed by |
| --- | --- | --- | --- | --- | --- |
| M1 | A | `main` (direct) | Repo skeleton, `.gitignore`, `requirements.txt`, `pyproject.toml`, `.env.ps1.example` | Before S1 | S1 |
| M2 | A | `a/lockfile` | `requirements.lock` | Dev A's own install works | S2 |
| M3 | B | `b/config-env` | `config.py`, `configs/default.yaml`, `00_check_env.py`, `conftest.py` | Your own check passes and Dev A has reviewed | S2 |
| M4 | A | `a/io-pose` | `io_video.py`, `pose.py`, `02_inventory.py`, `03_extract_skeletons.py` | Smoke extraction works | S4 |
| M5 | A | `a/labels` | `labels.py`, `test_labels.py` | Tests pass and the label check by eye looks right | S5 |
| M6 | B | `b/features` | `features.py`, `test_features.py` | Tests pass | S8 |
| M7 | B | `b/windows` | `windows.py`, `dataset.py`, `05_build_windows.py`, `test_windows.py` | Tests pass and the GMDCSA24 dry run works | S7 |
| M8 | A | `a/qc` | `04_qc_skeletons.py` | Runs on all core skeletons | S7 |
| M9 | B | `b/models` | `models.py`, `06_train.py`, `test_model.py` | Tests pass and one run trains | S10 |
| M10 | B | `b/events` | `calibrate.py`: `write_scores`, `alerts_from_scores` | Works on a toy score file | S9 |
| M11 | A | `a/eval` | `metrics.py`, `baseline.py`, `08_evaluate.py`, `test_metrics.py` | Pulled M6 and M10 and tests pass | S10 |
| M12 | A | `a/extras` | `configs/of_syn_subset.csv` and any extraction fixes | Before the extras are shared | S11 |
| M13 | A | `a/staged-tools` | `record_staged.py`, `label_tool.py`, staged support in inventory and labels | Before the recording session | S12 |
| M14 | B | `b/calibrate` | `07_calibrate.py` | Val curves look right | S14 |
| M15 | B | `b/live` | `09_live.py` | Runs at 10 fps or more | S15 |
| M16 | A and B | `a/docs`, `b/readme` | `DATA_CARD.md`, README sections | Before the final check | S15 |
| M17 | A | `main` | Tag `v1-hackathon` | Right after S15 | – |
| M18 | B | `b/contract-v2` | `default.yaml` v2 with the `occlusion` block, `windows.py` SCHEMA with `present` and `ctx`, `test_windows.py` | Right after S3 | M19 |
| M19 | A | `a/occlusion` | `occlusion.py`, `test_occlusion.py`, `reports/occlusion_realism.md` | Tests pass and the realism check matches OCCU | S16 |
| M20 | A | `a/tracker` | `tracker.py`, `test_tracker.py`, tracker rows in `08_evaluate.py` | Tests pass; scored on val | S17 |

## Step 2 – Data contracts (identical in both docs)

These formats let both devs code in parallel. Change one only if both agree, and bump `schema_version` in `configs/default.yaml`.

### Repo layout and owners

```
fallsense/
  configs/default.yaml           B
  src/fallsense/
    config.py                    B
    labels.py  io_video.py  pose.py  baseline.py  metrics.py  occlusion.py  tracker.py   A
    features.py  windows.py  dataset.py  models.py  calibrate.py  B
  scripts/
    00_check_env.py  05_build_windows.py  06_train.py  07_calibrate.py  09_live.py     B
    02_inventory.py  03_extract_skeletons.py  04_qc_skeletons.py  08_evaluate.py
    record_staged.py  label_tool.py                                               A
  tests/   test_labels.py test_metrics.py test_occlusion.py test_tracker.py (A)   test_features.py test_windows.py test_model.py (B)
  data/ -> junction to C:\fall-data (gitignored)
```

### Config values both sides rely on

`fps: 15`, `window: 30` frames (2 s), `stride_train: 8`, `stride_train_near_fall: 2`, `stride_eval: 3`, `keypoints: [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]` (nose, shoulders, elbows, wrists, hips, knees, ankles), `classes: [upright, falling, fallen, controlled_low]`, `datasets: [GMDCSA24, caucafall, le2i, mcfd, occu, of-syn]`, `test_only_datasets: [OOPS]`, `seed: 1337`, `on_floor_seconds: 20`.

### Contract 1 – skeleton file (Dev A's script writes, everyone reads)

`data/skeletons/{dataset}/{path}.npz`, one per clip; `{path}` is the OmniFall `path` column.

| Key | dtype, shape | Meaning |
| --- | --- | --- |
| `kp` | float32 (T, 33, 4) | x, y (0–1, image-normalised), z, visibility; NaN where no person |
| `world` | float32 (T, 33, 3) | world landmarks in metres, hip-centred; NaN where none |
| `detected` | bool (T,) | a pose was returned |
| `t_ms` | int64 (T,) | source timestamp of each kept frame |
| `fps` | int | 15 |
| `width`, `height` | int | source frame size |
| `dataset`, `path` | str | join keys to OmniFall labels |
| `subject`, `cam` | int | from OmniFall; -1 if unknown |
| `pose_model` | str | e.g. `pose_landmarker_full@0.10.x` |

No labels inside the .npz; labels are joined later so a mapping change never forces re-extraction.

### Contract 2 – skeleton index (Dev A writes)

`data/skeletons/index.csv`: `dataset, path, subject, cam, split, n_frames, duration_s, width, height, det_rate, det_rate_fall, det_rate_covered, det_rate_fall_covered, hidden_frac_fall, n_hidden_gaps, id_jumps, qc_pass, qc_reason` (v2 columns: see Occlusion handling); `split` is `train`, `val`, `test` or `staged_test`.

### Contract 3 – window arrays (you write)

`data/windows/{split}.npz`: `kp` float32 (N, 30, 13, 3) pixel x, y, visibility; `wh` int16 (N, 2); `y` int64 (N,) 0–3; `clip_id` int32 (row in `index.csv`); `end_frame` int32.

### Contract 4 – frame scores (you write for models, Dev A writes for the baseline; Dev A's evaluator reads)

`reports/scores/{model}/{split}.parquet`: `clip_id, end_frame, t_s, p_upright, p_falling, p_fallen, p_controlled_low`.

### Occlusion handling – schema_version 2 (identical in all three plans)

**Why.** A person who falls behind furniture disappears from the skeleton stream. On OCCU, MediaPipe finds the person in 99% of standing frames but in 3% of fallen frames (208 of 7,445), because they are fully hidden behind the bed. The skeleton file records "no person", but not *how* they vanished, and that context separates hidden falls from people leaving: of 60 disappearances during OCCU falls, 98% happened mid-frame (median 0.35 of the frame from the nearest border) with the hips dropping at 0.32 frame heights/s; disappearances outside falls happen near a border (median 0.11) while moving slowly (0.09). Background and the full list of ideas: `docs/occlusion-ideas.md`.

Three parts, used together:

1. **Disappearance context as model input.** Every window carries a per-frame "pose present" flag and four "last seen" values computed over the whole clip, so the model can tell "dropped fast mid-frame, then gone" (fallen) from "walked to the edge, then gone" (left the room).
2. **Disappearance tracker as a safety net.** A rule-based state machine (visible / left the frame / hidden after going down) raises CHECK when someone stays hidden after going down for `hidden_check_s`. It needs no training, runs offline and live, and is scored on its own and merged with the model's alerts.
3. **Simulated occlusion in training.** Visible falls *and* normal activity from every train set are occluded on the fly, gradually and realistically, so the model learns hidden falls from thousands of examples instead of OCCU's ~50 train falls. Nothing is copied to disk.

**Who owns what.** Dev A: `occlusion.py` (context, disappearance finder, simulation, live incremental state), `tracker.py`, the QC change, the realism check and the evaluation rows. Dev B: config v2, `windows.py` / `05_build_windows.py` changes, the four new features, the `dataset.py` hook, the `09_live.py` hook and experiment E7.

**`src/fallsense/occlusion.py` (Dev A; pure numpy; offline and live).** `kp` is (T, 13, 3) pixel x, y, visibility, the same array `features.py` takes.

| Function | Does |
| --- | --- |
| `last_seen_context(kp, present, wh, fps, cap_s) -> (T, 4)` | per frame: seconds since the pose was last present (0 while visible, capped at `cap_s`, `cap_s` if never seen); last visible hip height (y / H); last hip vertical speed (**body heights/s** over the last 5 present frames, + = down; body height = the tallest the person stood in the last 2 s, so near and far cameras read the same); last hip distance to the nearest border (0–0.5) |
| `LastSeenState` | the same four values updated one frame at a time, for `09_live.py` |
| `find_disappearances(kp, present, wh, fps, cfg) -> DataFrame` | every gap of at least 1 s after at least 5 present frames: start, end, edge distance, descent speed, and `kind` = `left` (lost within `edge_margin` of a border or not descending) or `hidden_down` (lost mid-frame while descending faster than `descent_speed`, in body heights/s) |
| `simulate_occlusion(kp, present, wh, rng, onset, mode, cfg) -> (kp, present)` | hides the person from frame `onset` on. `rise`: an occluder edge at a fixed image height; keypoints below it are lost, and the whole frame is lost once fewer than `min_visible_kp` remain, so ankles vanish first and the head last, as when sinking behind a bed. `box`: a fixed rectangle over 20–50% of the person's box (partial occlusion). Both add 3–8 frames of flicker at the boundary before the loss is complete. |

**Contract 2 v2 (index), Dev A.** Adds `det_rate_covered`, `det_rate_fall_covered`, `hidden_frac_fall` (share of falling/fallen frames inside a `hidden_down` gap), `n_hidden_gaps` and `id_jumps`. A frame counts as *covered* if the pose is detected or the frame lies inside a `hidden_down` gap; `det_rate_covered` also counts frames where the person is off-screen (before walking in or after walking out at a border, or in a gap that starts at a border), so an empty room is not a detection failure; `det_rate_fall_covered` does not. QC becomes `qc_pass = det_rate_covered >= 0.6 and (no fall frames or det_rate_fall_covered >= 0.5) and not id_switch` (more than 3 hip jumps over 25% of the diagonal between detections at most 2 frames apart), with `det_rate` and `det_rate_fall` still reported raw. OCCU train and val clips then pass on their merits instead of being dropped. *(Off-screen counting added 17 Oct after the first QC run: without it 105 of 192 MCFD clips and 15 Le2i train/val clips failed only because the person was out of that camera's view; with it MCFD keeps 118, and the fall check stays strict.)*

**Contract 3 v2 (windows), Dev B.** Adds two keys:

| Key | dtype, shape | Meaning |
| --- | --- | --- |
| `present` | bool (N, 30) | pose detected in that frame |
| `ctx` | float32 (N, 30, 4) | `last_seen_context` for the window's frames, computed over the whole clip (not just the window) |

The window drop rule changes from "over 40% of frames have no pose" to "over 40% of frames have no pose *and* no recent context", i.e. `(~present & (ctx[..., 0] >= cap_s)).mean() > 0.40`, so hidden-fallen windows are kept with their `fallen` label.

**Contract 4** is unchanged.

**Contract 5 (new): tracker alerts, Dev A writes, the evaluator reads.** `reports/alerts/tracker/{split}.parquet`: `clip_id` int32, `t_s` float32, `level` (always `CHECK`), `reason` (`hidden_down`).

**Features (Dev B).** `make_features(kp, wh, present, ctx)` returns (T, 51): the existing 47 plus the four context values (seconds since seen / `cap_s`, last hip height, last hip vertical speed, last edge distance). The existing "detected flag" feature is taken from `present`.

**Training-time simulation (Dev B calls Dev A's function).** In `dataset.py`, a `ClipBank` keeps every train clip's 13-keypoint pixel array and `present` in memory (about 50 MB for all train sets). With augmentation on, a window is occluded with probability `aug_p_fall` if a falling or fallen frame lies within `lookback_frames` before its end, otherwise with `aug_p_other`. To occlude: take the clip's last `lookback_frames` frames up to the window end, pick an onset inside them (for falls, between the fall start and the window end), call `simulate_occlusion`, recompute `last_seen_context` on that span, and keep the last 30 frames. The label does not change: a simulated hidden fall is still `fallen`, a person occluded while walking is still `upright`. Occluding normal activity matters as much as occluding falls; without it the model would learn "vanished means fallen". Val, test, staged_test and OOPS are never occluded artificially; OCCU test and the staged hidden-fall clips are the real-occlusion check.

**Realism check (Dev A).** `reports/occlusion_realism.md` compares simulated disappearances on Le2i and OF-Syn falls with real OCCU ones on edge distance, descent speed, gap length and keypoint loss order; adjust the simulation parameters until they match before Dev B trains E7. **Status (17 Oct): accepted as it is.** Edge distance, loss order and the false `hidden_down` rate match real OCCU; simulated falls vanish at a lower hip speed than real ones (36% classified `hidden_down` vs 83% for OCCU). E7 decides whether it helps; the report lists what to revisit if it does not.

**Config additions (`configs/default.yaml`, Dev B):**

```yaml
schema_version: 2
occlusion:
  ctx_cap_s: 20          # cap on seconds-since-seen (matches on_floor_seconds)
  edge_margin: 0.10      # lost within this of a border -> "left"
  descent_speed: 0.30    # hips dropping faster than this (body heights/s) when lost -> "hidden_down"
  hidden_check_s: 3.0    # tracker: hidden_down for this long -> CHECK; tuned on val
  min_visible_kp: 4      # simulation: fewer visible keypoints -> whole frame lost
  lookback_frames: 330   # simulation: clip history used (window + ctx_cap_s at 15 fps)
  aug_p_fall: 0.25       # simulation: probability for windows near a fall
  aug_p_other: 0.15      # simulation: probability for all other windows
```

`edge_margin` starts from the OCCU/Le2i measurement above. `descent_speed` is in **body heights/s** (changed from frame heights/s on 17 Oct: in frame units the same fall reads 3× faster on a close camera than a far one, 0.15 on MCFD vs 0.47 on GMDCSA24, while in body units falls read 0.67–0.99 everywhere); 0.30 was picked on train clips: it catches 83% of real OCCU hidden falls with 47 false `hidden_down` gaps outside falls, 43 of them on MCFD's poor-detection cameras (rechecked after gaps were debounced: a detector blip under 5 frames or a dropout of 1–2 frames no longer splits a disappearance) (`reports/occlusion_realism.md`). `hidden_check_s` is tuned on val only.

**Evaluation (Dev A).** `08_evaluate.py` adds rows for model + tracker and baseline + tracker, and reports OCCU test and the staged hidden-fall clips separately with and without each part. Experiment **E7** (best config + simulated occlusion vs the same config without it) answers "does simulated occlusion help?". **When E7 runs, Dev B writes its val results (with and without simulated occlusion: all of val, OCCU val, false alerts per hour elsewhere), the run names and the keep/drop decision into the "E7 outcome" section of `reports/occlusion_realism.md` and into `reports/experiments.csv`**, so the simulation can be revisited later.

**Merges and syncs added** (numbered after M17 and S15 so nothing else is renumbered):

| Id | Who | Branch / direction | What | When |
| --- | --- | --- | --- | --- |
| M18 | B | `b/contract-v2` | `default.yaml` v2 with the `occlusion` block; `windows.py` SCHEMA with `present` and `ctx`; updated `test_windows.py` | Right after S3 |
| M19 | A | `a/occlusion` | `occlusion.py`, `tests/test_occlusion.py`, `reports/occlusion_realism.md` | After M5, before S7 |
| M20 | A | `a/tracker` | `tracker.py`, `tests/test_tracker.py`, tracker rows in `08_evaluate.py` | With or before M11 |
| S16 | A → B | sync | `occlusion.py` merged; Dev B runs `pytest tests/test_occlusion.py` and reads the realism report | Before Dev B finishes M7 |
| S17 | A → B | sync | `tracker.py` merged; Dev B wires `DisappearanceTracker` and `LastSeenState` into `09_live.py` | Before M15 |

**Extra time (rough; replace with measured).** Dev A 5–7 h: `occlusion.py` and tests 2–3 h, realism check 1 h, tracker and tests 2–3 h, QC and evaluation changes about 1 h. Dev B 3–4 h: contract v2 about 1 h, features and windows 1.5–2 h, `ClipBank` and the augmentation hook 1–1.5 h, live hook 0.5 h. Training takes about 20–50% longer per run because the occluded examples are harder; on CPU that is minutes.

- [x] **SYNC S3 (both)** – read this section together, including the occlusion handling above; both say "agreed" to `schema_version: 2` before writing any more code. Dev B then merges the contract v2 changes (M18).
- [ ] **M18 (you)** – on branch `b/contract-v2`: bump `configs/default.yaml` to `schema_version: 2` and add the `occlusion` block; add `present` bool (N, 30) and `ctx` float32 (N, 30, 4) to `SCHEMA` in `windows.py` and to `save_windows`; extend `tests/test_windows.py` for the two keys. Pull request; Dev A reviews.

## Step 3 – Extract and share your three datasets

You run Dev A's scripts on your own videos (GMDCSA24, CAUCAFall, MCFD). `features.py` (Step 4) depends only on the contracts, so start it while you wait for S4 and while extraction runs.

- [ ] **SYNC S4 (A → B)** – Dev A says `io_video.py`, `pose.py`, `02_inventory.py` and `03_extract_skeletons.py` are merged and names the pose model (full or lite). Pull, then:
  - `python scripts\02_inventory.py --datasets GMDCSA24 caucafall mcfd` – read out clips and hours per dataset; no `missing` or `unreadable` flags.
  - `python scripts\03_extract_skeletons.py --datasets GMDCSA24 --limit 5` – open one .npz; `kp` is (T, 33, 4), `detected.mean()` above 0.8 on a walking clip.
  - Read the shapes and detection rate out loud. Tick when Dev A agrees.
- [ ] Full run: `python scripts\03_extract_skeletons.py --datasets GMDCSA24 caucafall mcfd` (estimate: about 120,000 frames, 15–25 min; MCFD is most of it). Keep the laptop on mains power; the script resumes if interrupted.
- [ ] If the ETA printed after 10 clips is over 2 hours, stop, drop MCFD to 4 of its 8 cameras with an edited inventory, and tell Dev A which cameras.
- [ ] Zip each dataset folder from `data\skeletons\` and upload to the shared folder `fallsense-data/skeletons/`, together with `reports\extract_log.csv`.
- [ ] **SYNC S6 (B → A)** – tell Dev A the three zips and the log are uploaded. Dev A unzips and reads out .npz counts per dataset; they match your log (GMDCSA24 160, CAUCAFall 100, MCFD as logged). Tick when the counts match out loud.
- [ ] Disk: keep the converted videos until S14 in case re-extraction is needed; you are well inside 30 GB.

## Step 4 – `features.py`

One function, `make_features(kp, wh, present, ctx) -> (T, 51) float32`, used by training, evaluation, Dev A's rule baseline and the live demo, so there is no train/serve mismatch. Input `kp` is (T, 13, 3): pixel x, pixel y, visibility for the 13 config keypoints (multiply normalised x by width and y by height first). `wh` is (width, height). `present` is (T,) bool and `ctx` is (T, 4) from Dev A's `occlusion.last_seen_context` (Contract 3 v2); until M19 lands, test with `ctx` computed by a stub that returns zeros while present.

- [ ] **Gap fill:** per keypoint, forward-fill NaN gaps of up to 5 frames (0.33 s); anything still missing becomes 0 with visibility 0.
- [ ] **Scale:** torso length L = distance from shoulder midpoint (keypoints 11, 12) to hip midpoint (23, 24), median over frames where all four have visibility above 0.5; fallback 0.25 × height; floor 0.05 × height.
- [ ] **Pose shape (26):** each keypoint's (x, y) minus that frame's hip midpoint, divided by L.
- [ ] **Visibility (13):** raw visibility per keypoint.
- [ ] **Derived (8), per frame:**
  - torso angle from vertical: angle of (shoulder mid − hip mid) against image up, divided by π (0 upright, 0.5 horizontal)
  - hip height = (ankle mid y − hip mid y) / L (about 2 standing, near 0 lying)
  - head height = (hip mid y − nose y) / L
  - hip vertical velocity = Δ(hip mid y) / Δt / L from un-centred coordinates (positive = moving down)
  - hip speed = |Δ hip mid| / Δt / L
  - log bounding-box aspect = log(width / height) of visible keypoints (Dev A's baseline uses this and hip height)
  - mean visibility
  - detected flag (0/1)
- [ ] **Disappearance context (4):** `ctx` columns as they are, except seconds-since-seen divided by `occlusion.ctx_cap_s` (0–1). The "detected flag" above comes from `present`.
- [ ] Clip every feature to \[−10, 10\]; assert no NaN leaves the function.
- [ ] Expose the feature names as `FEATURE_NAMES` (51 strings) so the baseline can index `log_bbox_aspect` and `hip_height` by name.
- [ ] `tests/test_features.py`: output (30, 51); a window whose last 10 frames are not present still returns finite values and carries the `ctx` values through; doubling all coordinates leaves shape features unchanged; a mirrored skeleton gives the same torso angle and hip height; 40% NaN frames still returns finite values; a synthetic standing-then-lying sequence moves torso angle from about 0 to about 0.5.
- [ ] Merge `features.py and its tests by pull request from branch b/features (M6); Dev A reviews and you both confirm main is green before S8`.
- [ ] **SYNC S8 (B → A)** – tell Dev A `features.py` is merged. Dev A pulls and runs `pytest tests/test_features.py`; it passes. Tick when Dev A confirms.

Feature z-scoring (mean and std over train windows only) happens in `06_train.py` and is saved as `feat_stats.npz` next to each model; evaluation and live code load it from there.

## Step 5 – Windows, splits and augmentation

Each example is a 2-second, 30-frame window labelled by what happens at its end. You need Dev A's labels (S5) to label windows and the QC'd index (S7) to build the real arrays; write and test the code on GMDCSA24 before S7 arrives.

- [ ] **SYNC S5 (A → B)** – Dev A says `labels.py` is merged. Pull, run `pytest tests/test_labels.py`, and look at the frame labels of one fall clip together. Tick when you both agree they line up with the motion.

### 5a. `src/fallsense/windows.py` – labelling rule

1. Drop the window if over 30% of frames are unlabelled, or over 40% of frames have no pose **and no recent context** (`~present & (ctx[:, 0] >= ctx_cap_s)`). A person hidden after a fall has recent context, so those windows are kept and labelled `fallen` (Contract 3 v2).
2. **falling** if at least 5 of the last 20 frames are labelled falling (falls are short; a majority rule would almost never fire).
3. Otherwise the most common label in the last 8 frames, ignoring unlabelled ones.

```python
def window_label(y4: np.ndarray, detected: np.ndarray) -> int | None:
    # v2: `detected` here means "present or recently seen": present | (ctx[:, 0] < ctx_cap_s)
    if (y4 < 0).mean() > 0.30 or (~detected).mean() > 0.40:
        return None
    if (y4[-20:] == 1).sum() >= 5:
        return 1
    tail = y4[-8:]
    tail = tail[tail >= 0]
    return int(np.bincount(tail, minlength=4).argmax()) if len(tail) else None
```

### 5b. `scripts/05_build_windows.py`

- [ ] Read `index.csv`. Train and val use `qc_pass` clips only; test, staged\_test and OOPS use every clip (keep a `qc_pass` column so results can be cut both ways).
- [ ] Per clip: load the .npz, take the 13 config keypoints, convert x and y to pixels with `width` and `height`, NaN where not detected.
- [ ] Per clip: `present = detected`; `ctx = occlusion.last_seen_context(kp, present, wh, fps, ctx_cap_s)` over the **whole clip**, then slice per window (a window alone cannot know someone vanished 8 s ago).
- [ ] Frame labels: `labels.segments_for(dataset, path)` → `clean_segments` → `frame_labels(segs, t_ms)` → `to4`.
- [ ] Window ends: train and val every 8 frames, plus every 2 frames wherever a falling frame lies within 1 s of the window end; test, staged\_test and OOPS every 3 frames.
- [ ] Apply `window_label`; save per Contract 3 v2 (including `present` and `ctx`) with `np.savez_compressed` to `data\windows\{train,val,test,oops}.npz` (OOPS kept as its own file so it is never mixed into test).
- [ ] Write `reports\windows_summary.csv`: counts by split × dataset × class. If falling is under 1% of train, lower the near-fall stride to 1.
- [ ] Assertions that stop the script: no `clip_id` in two splits; no `(dataset, subject)` in both train and test (except `subject == -1` and the MCFD single-actor note); no window crosses a clip boundary; every `y` is 0–3.
- [ ] Dry run on your GMDCSA24 skeletons before S7.
- [ ] **SYNC S7 (A → B)** – Dev A says core QC is done. Download Le2i skeletons, `index.csv` and `reports\qc\` from the shared folder; read out index row count and `qc_pass` counts per dataset; they match Dev A's. Tick when confirmed. Then run `05_build_windows.py` on all core sets.

### 5c. `src/fallsense/dataset.py` – augmentation

Augment raw keypoints, then call `make_features`.

**Simulated occlusion (occlusion handling part 3).** Add a `ClipBank` that loads every train clip's 13-keypoint pixel array and `present` once (about 50 MB). In `__getitem__`, with augmentation on, occlude the window with probability `aug_p_fall` if a falling or fallen frame lies within `lookback_frames` before its end, else `aug_p_other`: take the clip's last `lookback_frames` frames up to the window end, pick an onset (for falls, between fall start and window end; otherwise anywhere), call Dev A's `occlusion.simulate_occlusion` (mode `rise` or `box`, 50/50), recompute `last_seen_context` on that span, keep the last 30 frames. Keep the label. Never occlude val, test, staged\_test or OOPS.

- [ ] **SYNC S16 (A → B)** – Dev A says `occlusion.py` is merged. Pull, run `pytest tests/test_occlusion.py`, read `reports/occlusion_realism.md`, wire `simulate_occlusion` into `dataset.py`. Tick when Dev A agrees the hook is right.
- [ ] Check by eye: render 5 occluded training windows (skeleton on black) for falls and 5 for walking; the skeleton should sink and vanish for falls and vanish without dropping for walking.

```python
class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, stats, augment=False, seed=0):
        d = np.load(npz_path)
        self.kp, self.wh, self.y = d["kp"], d["wh"], d["y"]
        self.present, self.ctx = d["present"], d["ctx"]  # Contract 3 v2
        self.mu, self.sd, self.augment = stats["mu"], stats["sd"], augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        kp, present, ctx = self.kp[i].copy(), self.present[i].copy(), self.ctx[i].copy()
        if self.augment:
            kp, present, ctx = maybe_occlude(
                i, kp, present, ctx, self.rng
            )  # ClipBank + simulate_occlusion
            kp = augment_kp(kp, self.wh[i], self.rng)
        x = (make_features(kp, self.wh[i], present, ctx) - self.mu) / self.sd
        return torch.from_numpy(x.astype(np.float32)), int(self.y[i])
```

| Augmentation | Probability | Range | Simulates |
| --- | --- | --- | --- |
| Mirror left–right (flip x, swap L/R pairs) | 0.5 | – | Falls to either side |
| In-plane rotation about window hip midpoint | 0.5 | ±10° | Camera roll |
| Scale about hip midpoint | 0.5 | 0.85–1.15 | Distance to camera |
| Translation | 0.5 | ±10% of frame | Position in room |
| Coordinate jitter | 1.0 | σ = 1% of torso length | Pose noise |
| Joint dropout (NaN) | 0.05 per joint-frame | – | Furniture occlusion |
| Whole-frame dropout | 0.03 per frame | – | Lost detection |
| Simulated occlusion (`simulate_occlusion`, needs the `ClipBank`) | `aug_p_fall` near falls, `aug_p_other` elsewhere | `rise` or `box` | Falls behind furniture; people walking behind it |
| Speed change (resample in time, anchored at last frame) | 0.3 | 0.8–1.2× | Slower, frailer movement |

### 5d. `tests/test_windows.py`

- [ ] A window ending 0.5 s into a 1 s fall is falling; 3 s after the fall with the person on the floor is fallen; a sit-down is controlled\_low; a clip under 30 frames gives zero windows; the split assertions fire on a deliberately leaked toy index. Merge windows.py, dataset.py, 05\_build\_windows.py and these tests by pull request from branch b/windows (M7) before S7; Dev A reviews.

Later: after **S11** (extras) and **S13** (staged), rerun `05_build_windows.py`; those syncs are in Steps 6 and 8.

## Step 6 – Models, training, experiments and event logic

Training is cheap (minutes per run on CPU), so run every experiment with three seeds. Dev A's evaluator needs your event logic and score writer (S9) before it can compare E0 and E1 (S10). Train on your laptop while Dev A's keeps extracting the extras.

### 6a. `src/fallsense/models.py`

```python
class FallGRU(nn.Module):
    def __init__(self, f_in=51, hidden=64, n_cls=4, layers=2, dropout=0.2):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(f_in, hidden), nn.ReLU(), nn.LayerNorm(hidden)
        )
        self.gru = nn.GRU(
            hidden, hidden, num_layers=layers, batch_first=True, dropout=dropout
        )
        self.head = nn.Linear(hidden, n_cls)

    def forward(self, x):  # x: (B, 30, 51)
        h, _ = self.gru(self.inp(x))
        return self.head(h[:, -1])  # logits from the last time step


class FallTCN(nn.Module):
    """3 dilated 1D conv blocks (dilation 1, 2, 4), kernel 5, 64 channels,
    BatchNorm + ReLU + Dropout(0.1), then concat(mean over time, last step) -> Linear."""
```

- [ ] `tests/test_model.py`: both models map (8, 30, 51) to (8, 4); a 50-step overfit on 64 windows drives loss near zero. Merge models.py, 06\_train.py and this test by pull request from branch b/models (M9) once one run trains.

### 6b. `scripts/06_train.py`

- [ ] Flags: `--model gru|tcn`, `--train-sets`, `--augment/--no-augment`, `--seed`, `--out models\<run-name>`.
- [ ] Seed `random`, `numpy`, `torch`; `torch.set_num_threads(os.cpu_count() - 1)`.
- [ ] Compute feature `mu`/`sd` over train windows (no augmentation); save `feat_stats.npz`.
- [ ] Loss `CrossEntropyLoss(weight=w, label_smoothing=0.05)`, `w_c = sqrt(N / (4 * n_c))` renormalised to mean 1 (weights, not an oversampling sampler).
- [ ] AdamW lr 2e-3, weight decay 1e-4, batch 256, cosine decay over at most 40 epochs, gradient clip 1.0.
- [ ] Select on val each epoch: falling-vs-rest average precision, tie-broken by 4-class macro F1; early stop after 8 epochs without improvement.
- [ ] DataLoader `num_workers=4, persistent_workers=True`; the whole script under `if __name__ == "__main__":` (Windows workers crash otherwise).
- [ ] Save `model.pt`, `feat_stats.npz`, config copy, `train_log.csv`, `val_logits.npy`, confusion matrix PNG.
- [ ] Print seconds per epoch after epoch 1 and an ETA. If an epoch passes 2 minutes, precompute 3 augmented copies of train features to `data\windows\train_aug.npz` and train from that (about 10× faster).

| Train set | Approx. train windows | Per epoch | One run | E-row, 3 seeds |
| --- | --- | --- | --- | --- |
| Core staged | \~20,000 | 5–15 s | 3–10 min | 10–30 min |
| Core + OCCU | \~23,000 | 5–20 s | 4–12 min | 12–36 min |
| Core + OCCU + OF-Syn subset (5,335 clips, all elderly) | \~59,000 | 18–48 s | 12–30 min | 36–90 min |

### 6c. Event logic and score writer in `src/fallsense/calibrate.py` (Dev A is waiting on this)

- [ ] `write_scores(model_dir, split_npz, out_parquet)`: run the model on every window, apply temperature if present, write Contract 4 rows (`clip_id, end_frame, t_s, p_upright, p_falling, p_fallen, p_controlled_low`).
- [ ] `alerts_from_scores(scores, thresholds) -> DataFrame[clip_id, t_s, level]` per clip in time order:
  - fall score `s` = `p_falling`, max over the last 3 evaluation windows (0.6 s)
  - `s >= t_high` → FALL; `t_low <= s < t_high` → CHECK
  - `p_fallen > 0.5` continuously for `on_floor_seconds` → ON FLOOR (use 5 s on public clips, 20 s in the demo; read from `thresholds.yaml`)
  - 30 s refractory after any alert
- [ ] Merge calibrate.py (write\_scores, alerts\_from\_scores) by pull request from branch b/events (M10); Dev A reviews, because their evaluator and tests need it on `main`.
- [ ] **SYNC S9 (B → A)** – tell Dev A both functions are merged. Dev A runs `tests/test_metrics.py` against them; it passes. Tick when Dev A confirms.

### 6d. Experiments (one row each in `reports\experiments.csv`, 3 seeds)

| Id | Model | Train data | Augment | Question |
| --- | --- | --- | --- | --- |
| E0 | Rule baseline (Dev A) | core | – | What does a hand rule achieve? |
| E1 | GRU | core | no | Does learning beat the rule? |
| E2 | GRU | core | yes | Does augmentation help on held-out people? |
| E3 | TCN | core | yes | GRU or TCN? |
| E4 | best of E2/E3 | core + OCCU | yes | Does occlusion data help? |
| E5 | best of E2/E3 | core + OCCU + OF-Syn subset | yes | Does synthetic diversity help, especially older and heavier bodies? |
| E6 | best config | all except CAUCAFall | yes | How much does accuracy drop on an unseen home (CAUCAFall test)? |
| E7 | best config | same as the best of E4/E5 | yes + simulated occlusion | Does simulated occlusion help on hidden falls (OCCU val/test, staged hidden falls) without more false alerts elsewhere? |

"Core" = GMDCSA24, CAUCAFall, Le2i, MCFD.

- [ ] Train E1; write its val scores with `write_scores` (temperature 1, temporary thresholds 0.5/0.5).
- [ ] **SYNC S10 (A → B)** – Dev A says `08_evaluate.py` and E0 val results are merged. Run `python scripts\08_evaluate.py --model <E1 run> --split val`; both read out sensitivity and false FALL alerts per hour for E0 and E1. Tick when the numbers look sensible to both.
- [ ] Train E2 and E3; compare on val with the evaluator.
- [ ] **SYNC S11 (A → B)** – Dev A says OCCU, OF-Syn subset and OOPS skeletons are QC'd and uploaded with a new `index.csv`. Download, rerun `05_build_windows.py`, and read out window counts for OCCU, OF-Syn and OOPS. Tick when Dev A agrees.
- [ ] Train E4, E5, E6. Keep an addition only if it improves val; pick the final configuration on val only.
- [ ] Train E7 (best of E4/E5 plus simulated occlusion) against the same config without it. Write the val results (all of val, OCCU val, false alerts per hour elsewhere), both run names and the keep/drop decision into the "E7 outcome" section of `reports/occlusion_realism.md` and a row in `reports/experiments.csv`. If E7 does not help, say so there; the report lists what to revisit in the simulation.

## Step 7 – Calibration and CHECK thresholds

Turn the chosen model's probabilities into FALL, CHECK or silent using two thresholds picked on val only. The CHECK band is the measurable "knows when to hand over" behaviour for the pitch.

### `scripts/07_calibrate.py`

- [ ] **Temperature scaling:** fit one scalar T on `val_logits.npy` by minimising negative log-likelihood (`torch.optim.LBFGS`, start T = 1.0, 50 iterations). Report expected calibration error before and after (15 bins).
- [ ] Rewrite val scores with `write_scores` using T.
- [ ] **Threshold search on val:** sweep `t_high` and `t_low` over 0.05–0.95 in steps of 0.05, using `alerts_from_scores` and Dev A's metrics.
  - `t_high` = lowest value giving at most 0.5 false FALL alerts per hour of normal activity
  - `t_low` = highest value keeping event sensitivity (FALL or CHECK) at 95% or above; if none reaches 95%, take the best and note it
- [ ] Save `temperature`, `t_high`, `t_low`, `on_floor_seconds_eval: 5`, `on_floor_seconds_demo: 20` to `models\<run>\thresholds.yaml`.
- [ ] Run `08_evaluate.py --model <run> --split val` and check the coverage–error curve looks right before freezing. Merge 07\_calibrate.py by pull request from branch b/calibrate (M14) before S14.

Rule: test, staged\_test and OOPS are not opened until the model is frozen at S14.

## Step 8 – Staged session, freeze, live demo and done check

### 8a. Staged recording session (Dev A runs it; you act and spot)

- [ ] Sign the consent form before recording.
- [ ] Safety: falls only onto crash mats or stacked yoga mats, slow and controlled, never near stairs or furniture edges; one person always spots.
- [ ] Act your share of the clip list in Dev A's Step 6 (falls forward, backward, sideways, off a chair, behind a chair; sit, floor-sit, kneel, lie down, tie shoe, walk past) across both camera positions and both lighting levels.
- [ ] **SYNC S12 (both)** – walk Dev A's coverage table together (actions × cameras × lighting × actors); record any gaps before packing up. Tick when both agree coverage is complete.
- [ ] Independently check the start and end of every fall segment in Dev A's `labels.csv`; correct anything off by more than 0.2 s and tell Dev A.
- [ ] **SYNC S13 (A → B)** – Dev A uploads staged skeletons, labels and the updated `index.csv`. Confirm out loud that you double-checked the fall times, run `05_build_windows.py` to build `staged_test.npz`, and read out clip and window counts. Tick when Dev A agrees.

### 8b. Freeze and hand over

- [ ] Freeze the chosen run: no further changes to model, temperature or thresholds.
- [ ] Write scores once with `write_scores` for `test`, `staged_test` and `oops`.
- [ ] Upload `models\<run>\model.pt`, `feat_stats.npz`, `thresholds.yaml` and the three score files to the shared folder.
- [ ] **SYNC S14 (B → A)** – tell Dev A the run name and that the files are uploaded. Dev A confirms they are there, runs the final evaluation once per split, and you review the reports together (headline numbers with sample sizes and intervals, coverage–error curve, OF-Syn age/BMI subgroup table). Tick when both have reviewed.

### 8c. `scripts/09_live.py`

- [ ] `cv2.VideoCapture(0)` at 640×480; one `PoseExtractor` in VIDEO mode with timestamps from `time.monotonic()` in ms (strictly increasing).
- [ ] Keep frames at 15 fps by time; push the 13-keypoint pixel coordinates into a 30-frame ring buffer.
- [ ] Every frame: update `occlusion.LastSeenState` and Dev A's `DisappearanceTracker` (S17); keep `present` and `ctx` in the ring buffer alongside the keypoints.
- [ ] Every 3rd frame: `make_features(kp, wh, present, ctx)` → z-score with `feat_stats.npz` → model → `softmax(logits / T)` → the same `alerts_from_scores` logic with `on_floor_seconds_demo` (20 s). A tracker CHECK is shown and printed like a model CHECK, with `"source": "tracker"`.
- [ ] Display a skeleton on a black canvas (never the camera image), the four probabilities as bars, the current level (silent, CHECK, FALL) and an ON FLOOR timer.
- [ ] On every alert print one JSON line: `{"room": "demo", "time": ..., "level": "CHECK", "score": 0.58, "source": "pose", "model": "<run>"}` – the interface the carer app will consume later.
- [ ] Print processing fps once a second; it must stay at 10 or above. If not: switch to `pose_landmarker_lite.task`, or run pose on every 2nd frame. Merge 09\_live.py by pull request from branch b/live (M15) once it holds 10 fps, before S15.

### 8d. README (your part)

- [ ] Model section: features, architecture, training settings, experiments table with val results, chosen run, temperature and thresholds, how to run `09_live.py`. Merge by pull request from branch b/readme (M16) before S15; after S15 passes, Dev A tags main as v1-hackathon (M17).

### Your risks to watch

| Risk | Early sign | Fallback |
| --- | --- | --- |
| Too few falling windows | falling under 1% in `windows_summary.csv` | Near-fall stride 1; check class weights |
| Model learns each dataset's camera, not falls | E6 drops sharply versus E2 | More rotation and scale augmentation; add OCCU and OF-Syn; say so in the pitch |
| Epochs slow with OF-Syn | over 2 min per epoch | Train from precomputed `train_aug.npz` |
| Windows DataLoader crashes | error on worker start | Script body under `if __name__ == "__main__":`; or `num_workers=0` |
| Live demo under 10 fps | fps print | Lite pose model; pose every 2nd frame |
| Model learns "vanished means fallen" | false FALLs when someone walks out of view on val | Raise `aug_p_other` so more normal activity is occluded; check the context features are in the input |
| Simulated occlusion hurts accuracy on visible falls | E7 worse than its base on non-OCCU val | Lower `aug_p_fall`; keep E7 only if hidden-fall gains outweigh it, decided on val |

### Definition of done

- [ ] `data\skeletons\` has one .npz per clip for every dataset used, all through QC (Dev A)
- [ ] `data\windows\` has train, val, test, staged\_test and OOPS arrays from person-disjoint splits
- [ ] Baseline and GRU scored by the same `08_evaluate.py` on the same windows and events
- [ ] Model, temperature and thresholds saved in `models\<run>\`
- [ ] `09_live.py` shows skeleton and alert level on a webcam at 10 fps or more
- [ ] `DATA_CARD.md` complete (Dev A)
- [ ] **SYNC S15 (both)** – demo `09_live.py` to Dev A (skeleton on black, 10+ fps); walk this checklist together and tick each item out loud.
