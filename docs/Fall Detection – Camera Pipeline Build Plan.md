# Fall Detection – Camera Pipeline Build Plan

Oct 17, 2026 · @K

## Goal and scope

The end state is a calibrated skeleton-sequence classifier that labels 2-second windows as upright, falling, fallen or lying/sitting on purpose, beats a rule baseline on held-out people, and runs live on a laptop webcam. Audio, the carer app, escalation and the manager log are out of scope for this plan.

The pipeline in one line: public fall videos → cleaned 15 fps clips → MediaPipe skeletons (.npz) → normalised features → 30-frame windows → GRU → temperature-scaled probabilities → FALL / CHECK / silent thresholds. Raw video is deleted once skeletons are extracted and QC'd.

Definition of done:

- [ ] `data/skeletons/` holds one .npz per clip for every dataset we use, all passing QC
- [ ] `data/windows/` holds train / val / test window arrays built from person-disjoint splits
- [ ] Rule baseline and GRU both scored by the same `evaluate.py` on the same test windows and events
- [ ] GRU checkpoint, temperature and two thresholds saved in `models/gru-v1/`
- [ ] `live.py` shows the skeleton and the alert level on a webcam feed at 10 fps or better on CPU
- [ ] `DATA_CARD.md` lists every dataset, licence, clip count and what was excluded

## Who owns what

Dev A owns **data in** (cleaning, labelling and skeletonising video, plus the staged test set and the scorer). Dev B owns **model out** (repo, features, windows, training, calibration, live demo). All public video was downloaded overnight: the heavy sets on A's laptop (200 GB cap), the light ones on B's (30 GB cap). Each laptop extracts skeletons from its own videos, so only small skeleton files move between machines.

| Area | Dev A (data in) | Dev B (model out) |
| --- | --- | --- |
| First thing | Morning download check on A's laptop; project venv | Morning download check on B's laptop; creates repo, venv, lockfile, config, test runner |
| Videos on own laptop (downloaded overnight) | Le2i, OCCU, OF-Syn, OOPS (test only) | GMDCSA24, CAUCAFall, MCFD |
| Skeletons extracted on own laptop | Le2i, OCCU, OF-Syn 5,335-clip subset (all 2,000 elderly clips), OOPS | GMDCSA24, CAUCAFall, MCFD |
| Modules written | `labels.py`, `io_video.py`, `pose.py`, `occlusion.py`, `tracker.py` | `features.py`, `windows.py`, `dataset.py`, `models.py`, `calibrate.py` |
| Scripts written | `02_inventory.py`, `03_extract_skeletons.py`, `04_qc_skeletons.py`, `08_evaluate.py`, `baseline.py`, `record_staged.py`, `label_tool.py` | `00_check_env.py`, `05_build_windows.py`, `06_train.py`, `07_calibrate.py`, `09_live.py` |
| Data deliverables | `data/skeletons/**.npz`, `index.csv`, `qc.csv`, staged test set, `DATA_CARD.md` | `data/windows/*.npz`, `models/gru-v1/`, `reports/experiments.csv` |
| Tests | `test_labels.py`, `test_metrics.py`, `test_occlusion.py`, `test_tracker.py` | `test_features.py`, `test_windows.py`, `test_model.py` |

Handoffs that unblock the other person, in order (no deadlines; each one starts the next piece of work):

1. **A→B** – `pose.py` and the skeleton .npz schema frozen; B starts extracting GMDCSA24, CAUCAFall and MCFD on B's laptop, A starts Le2i and OCCU on A's.
2. **A→B** – `labels.py` merged; B builds the first windows from GMDCSA24 skeletons and tests `features.py` on real data.
3. **A↔B** – all core skeletons (GMDCSA24, CAUCAFall, Le2i, MCFD) QC'd and in the shared folder; `index.csv` final for core sets.
4. **A→B** – `08_evaluate.py` and rule baseline scored on val; B's first GRU is compared against it.
5. **A→B** – OCCU, OF-Syn subset and OOPS skeletons QC'd and shared (A's laptop keeps extracting these while B trains).
6. **A→B** – staged test set skeletons and labels shared.
7. **B→team** – calibrated model frozen, final report written.

## Stack and environment setup

Run everything natively on Windows (or macOS) with Python 3.11 managed by `uv`, not in WSL: webcams are awkward to pass into WSL, and `uv` avoids the pip/PATH clashes a system-wide Python 3.14 causes. MediaPipe wheels lag new Python releases, so do not use 3.13 or 3.14 for this project.

| Layer | Choice | Why |
| --- | --- | --- |
| Python | 3.11 via `uv` | Supported by MediaPipe, PyTorch and the `omnifall` package (needs 3.10+) |
| Dataset tooling | `omnifall[video]` + `ffmpeg` | Downloads, converts and verifies the public datasets into one layout with one label set |
| Video I/O | `opencv-python` | Frame reading and webcam capture |
| Pose | `mediapipe` Pose Landmarker (Tasks API), `pose_landmarker_full.task` | 33 keypoints, CPU real time, Apache 2.0 |
| Arrays and tables | `numpy`, `pandas`, `pyarrow` | .npz skeletons, CSV/parquet indexes |
| Model | `torch` (CPU build) | Small GRU / 1D-CNN; no GPU needed |
| Metrics and plots | `scikit-learn`, `matplotlib` | Confusion matrices, PR and coverage curves |
| Config and CLI | `pyyaml`, `typer`, `tqdm` | One config file, scripts with flags, progress bars |
| Quality | `pytest`, `ruff` | Fast tests for labels, features and windows; one formatter |

### Morning setup (both devs; tools and datasets were installed overnight)

1. Already done overnight on both laptops: `uv`, `ffmpeg`, the `omnifall` command (installed with `uv tool install "omnifall[video]" --python 3.11 --with requests`), and the datasets in `C:\fall-data\cache\videos\<dataset>\video\`. Check every download window ended with a green `<dataset> DONE`.
2. Verify each dataset on that laptop with `omnifall verify <dataset>`: A runs it for `le2i`, `occu`, `of-syn`, `OOPS`; B for `GMDCSA24`, `caucafall`, `mcfd`. Any failure → the fallbacks in the Datasets section.
3. Disk: B's archives were deleted as each dataset finished, so B sits well under its 30 GB cap. On A, once every verify passes, delete `C:\fall-data\downloads` to free about 30 GB.
4. Dev B only: create the repo (layout in the next section), push it, add Dev A as a collaborator.
5. Clone to `C:\code\fallsense`, then from the repo root:
   - `uv venv --python 3.11 .venv`
   - `.venv\Scripts\activate`
   - `uv pip install -r requirements.txt`
   - `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`
6. Dev B, once it installs cleanly: `uv pip freeze > requirements.lock` and commit it. Dev A then runs `uv pip sync requirements.lock` so both machines match exactly.
7. Link the data folder into the repo (gitignored): `New-Item -ItemType Junction -Path data -Target C:\fall-data`.
8. Save environment variables once, then close and reopen every terminal:
   - `setx FALL_DATA "C:\fall-data"`
   - `setx OMNIFALL_CACHE_DIR "C:\fall-data\cache"`
   - `setx OMNIFALL_DOWNLOAD_DIR "C:\fall-data\downloads"`
   - `setx OMNIFALL_ROOT "C:\fall-data\cache\videos"` (only valid now the folder exists; it must point at the folder holding `<dataset>\video`)
9. Download the pose models (use `curl.exe`; plain `curl` in Windows PowerShell is a different command):
   - `New-Item -ItemType Directory -Force models\pose`
   - `curl.exe -L -o models\pose\pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task`
   - the same for `pose_landmarker_lite` (swap `full` for `lite` in both places). If a link 404s, copy the current one from the Models section of the [Pose Landmarker overview](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker).
10. Run `python scripts\00_check_env.py`; every line must print OK before anyone continues.

`requirements.txt`:

```
omnifall[video]>=0.2
requests
mediapipe>=0.10.14
opencv-python>=4.9
numpy
pandas
pyarrow
scikit-learn
matplotlib
pyyaml
typer
tqdm
pytest
ruff
```

`scripts/00_check_env.py` checks, in order: Python is 3.11; `import mediapipe, cv2, torch, omnifall` succeed; `shutil.which("ffmpeg")` is not None; the four env vars are set, OMNIFALL\_ROOT exists, and the drive has at least 10 GB free for skeletons and models (`shutil.disk_usage`); the .task file loads in a `PoseLandmarker`; one webcam frame can be read with `cv2.VideoCapture(0)` and returns a pose on a person in view; `torch.get_num_threads()` is printed so you know how many CPU cores training will use.

## Repo layout, config and data contracts

The two devs work in parallel only because the file formats below are fixed before any code is written; change one only by agreement and a version bump in `configs/default.yaml`.

### Layout

```
fallsense/
  README.md  DATA_CARD.md  requirements.txt  requirements.lock  .gitignore  .env.ps1.example
  configs/default.yaml
  src/fallsense/
    __init__.py  config.py
    labels.py      # A: OmniFall 16 classes -> our 4, segments -> per-frame labels
    io_video.py    # A: timestamp-based 15 fps frame reader
    pose.py        # A: MediaPipe wrapper, one class used offline AND live
    features.py    # B: keypoint normalisation + derived features (single source of truth)
    windows.py     # B: 30-frame windows + window labelling rule
    dataset.py     # B: torch Dataset with augmentation
    models.py      # B: GRU and 1D-CNN
    baseline.py    # A: rule baseline
    calibrate.py   # B: temperature scaling + threshold search
    metrics.py     # A: window and event metrics, bootstrap CIs
    occlusion.py   # A: last-seen context, disappearance finder, occlusion simulation
    tracker.py     # A: disappearance tracker (CHECK when hidden after going down)
  scripts/
    00_check_env.py  01_prepare_datasets.ps1  02_inventory.py  03_extract_skeletons.py
    04_qc_skeletons.py  05_build_windows.py  06_train.py  07_calibrate.py
    08_evaluate.py  09_live.py  record_staged.py  label_tool.py
  tests/
  models/pose/  models/gru-v1/
  reports/
  data/  -> symlink or junction to C:\fall-data (gitignored)
```

`.gitignore` must contain `data/`, `*.mp4`, `*.avi`, `*.npz`, `.venv/`, `models/pose/*.task`. Model checkpoints are under 1 MB and can be committed.

### configs/default.yaml

```yaml
schema_version: 2          # v2 adds the `occlusion` block (see Occlusion handling below)
fps: 15                    # every clip is resampled to this
window: 30                 # frames = 2.0 s
stride_train: 8            # ~0.5 s
stride_train_near_fall: 2  # denser windows within 2 s of a fall segment
stride_eval: 3             # 0.2 s, for latency measurement
pose_model: models/pose/pose_landmarker_full.task
pose: {num_poses: 1, min_det: 0.5, min_presence: 0.5, min_track: 0.5}
keypoints: [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # nose, shoulders, elbows, wrists, hips, knees, ankles
classes: [upright, falling, fallen, controlled_low]
datasets: [GMDCSA24, caucafall, le2i, mcfd, occu, of-syn]
test_only_datasets: [OOPS]
split_config_suffix: "-cs"   # OmniFall cross-subject splits
seed: 1337
on_floor_seconds: 20       # demo value; evaluated at 5 s on public clips
```

### Contract 1: skeleton file (A writes, everyone reads)

One file per clip at `data/skeletons/{dataset}/{path}.npz`, where `{path}` is the OmniFall `path` column (it may contain sub-folders).

| Key | dtype, shape | Meaning |
| --- | --- | --- |
| `kp` | float32 (T, 33, 4) | x, y (0–1, image-normalised), z, visibility; NaN where no person found |
| `world` | float32 (T, 33, 3) | MediaPipe world landmarks in metres, hip-centred; NaN where none |
| `detected` | bool (T,) | a pose was returned for this frame |
| `t_ms` | int64 (T,) | source timestamp of each kept frame |
| `fps` | int | always 15 |
| `width`, `height` | int | source frame size in pixels (needed to undo x/y normalisation) |
| `dataset`, `path` | str | join keys back to OmniFall labels |
| `subject`, `cam` | int | from OmniFall; -1 if unknown |
| `pose_model` | str | e.g. `pose_landmarker_full@0.10.x` |

Per-frame labels are **not** stored in the .npz, so a change to the label mapping never forces re-extraction.

### Contract 2: skeleton index (A writes)

`data/skeletons/index.csv`, one row per clip: `dataset, path, subject, cam, split, n_frames, duration_s, width, height, det_rate, det_rate_fall, det_rate_covered, det_rate_fall_covered, hidden_frac_fall, n_hidden_gaps, id_jumps, qc_pass, qc_reason` (v2 columns: see Occlusion handling). `split` is `train`, `val`, `test` or `staged_test`.

### Contract 3: window arrays (B writes)

`data/windows/{split}.npz`:

| Key | dtype, shape | Meaning |
| --- | --- | --- |
| `kp` | float32 (N, 30, 13, 3) | pixel-space x, y and visibility for the 13 keypoints, before normalisation (so augmentation can run on it) |
| `wh` | int16 (N, 2) | width, height of the source clip |
| `y` | int64 (N,) | window label, 0–3 |
| `clip_id` | int32 (N,) | row number in `index.csv` |
| `end_frame` | int32 (N,) | index of the window's last frame in its clip |

### Contract 4: frame scores (B writes, A's evaluator reads)

`reports/scores/{model}/{split}.parquet`: `clip_id, end_frame, t_s, p_upright, p_falling, p_fallen, p_controlled_low`, one row per evaluation window. The rule baseline writes the same file with 0/1 values, so `08_evaluate.py` scores both identically.

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

### Git habits

Branch per task (`a/labels`, `b/features`), small PRs, the other dev reviews in under 10 minutes, `.githooks/pre-push` runs `ruff format --check` and `ruff check` on every push (a few seconds; turn it on once per clone with `git config core.hooksPath .githooks`); run `pytest -q` yourself before opening a PR. Type hints are welcome but not enforced. Never commit anything under `data/`.

## Datasets: what, where, licence

We train on six staged sources plus a 5,335-clip subset of OF-Syn weighted to elderly bodies, and test on OOPS genuine accidents; all of it was downloaded overnight with the [`omnifall` package](https://pypi.org/project/omnifall/), split across the two laptops by how much compute each set needs. `omnifall` converts every release to one layout (`C:\fall-data\cache\videos\<dataset>\video\<path>.mp4`) and checks it against the frame-accurate OmniFall labels, so nobody writes per-dataset parsers.

| Dataset | Laptop | Original source | Download | Video | Licence (verify with `omnifall sources <ds>`) | Use |
| --- | --- | --- | --- | --- | --- | --- |
| [GMDCSA24](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos) | B | GitHub / Zenodo | 1.0 GiB | 160 videos, 0.36 h, 4 actors, 3 homes, laptop webcam | MIT | Train / val / test; first dataset through the pipeline |
| [CAUCAFall](https://data.mendeley.com/datasets/7w7fccy7ky/4) | B | Mendeley Data | 7.8 GiB | 100 videos, 0.28 h, 10 subjects, home, includes night IR and occlusion | CC BY 4.0 | Train / val / test; closest to a care-home room |
| [MCFD](https://www.iro.umontreal.ca/~labimage/Dataset/) | B | Université de Montréal | 3.5 GiB | 24 scenarios × 8 cameras, 1.6 h across all views | Citation requested; check terms | Train / val; high corner cameras like a corridor mount |
| [Le2i](https://search-data.ubfc.fr/imvia/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html) | A | Université de Bourgogne (ImViA) | 8.9 GiB | 190 videos, 0.79 h, 4 rooms, 320×240 at 25 fps | Research use; check terms | Train / val / test |
| [OCCU](https://doi.org/10.5281/zenodo.15494102) | A | Zenodo | 10.1 GiB | 2 views, 0.48 h, falls partly hidden by furniture | Check terms | Train / val / test; occlusion |
| [OF-Syn](https://huggingface.co/datasets/simplexsigil2/omnifall) | A | Hugging Face | 9.0 GiB | 12,000 synthetic clips, 16.9 h, with age group, BMI band and camera metadata; we extract a 5,335-clip subset with all 2,000 elderly clips | Non-commercial; check card | Train, plus subgroup test (older, heavier bodies) |
| [OOPS-Fall](https://oops.cs.columbia.edu/data/) | A | Columbia, via `omnifall` | streamed \~45 GiB, kept 2.6 GiB | 818 genuine accident clips, 2.65 h | Non-commercial; check terms | Out-of-domain test only, never trained on |
| [OmniFall labels](https://huggingface.co/datasets/simplexsigil2/omnifall) | both | Hugging Face | a few MB | Segment labels and cross-subject splits for all of the above | CC BY-NC-SA 4.0 | Labels and splits |

Skipped: UP-Fall (1,118 archives and a slow frame-to-video rebuild), CMDFall (access by email only) and EDF (14.9 GiB, same lab and setup as OCCU without the occlusion). OOPS and OF-Syn use classes 10–15 as well; `labels.py` maps them to upright. Because the OmniFall labels are non-commercial, note in the README that a commercial model would need its own labels.

OmniFall classes used by the staged sets ([LABELS.md](https://huggingface.co/datasets/simplexsigil2/omnifall/blob/main/LABELS.md)): 0 walk, 1 fall, 2 fallen, 3 sit\_down, 4 sitting, 5 lie\_down, 6 lying, 7 stand\_up, 8 standing, 9 other. Label rows are `path, label, start, end, subject, cam, dataset` with times in seconds.

### Downloads (done overnight) and morning checks

Each laptop ran `omnifall prepare <dataset> --workers 4` overnight (up to 3 retries per dataset). Before writing any code, each dev checks their own laptop:

1. `omnifall verify <dataset>` for each dataset on that laptop must report no missing files.
2. **Le2i failed** (its site has a bot check): `omnifall sources le2i` for the exact archive name, download it in a browser into `C:\fall-data\downloads` without renaming, then `omnifall prepare le2i --workers 4`.
3. **CAUCAFall failed**: on the Mendeley page click Download All, then `omnifall prepare caucafall --archive "$HOME\Downloads\Dataset CAUCAFall.zip"`.
4. **Anything else failed or stopped part-way**: rerun `omnifall prepare <dataset> --workers 4`; it reuses archives already downloaded.
5. `omnifall cite <dataset>` for each, pasted into `DATA_CARD.md` with the terms printed by `omnifall sources <dataset>`.

OF-Syn subset (Dev A): all 12,000 clips are on disk, but extracting all of them would take 1.7–3.4 hours, so extract a 5,335-clip subset: every elderly clip (the population this product is for) plus a third of each other age group.

1. `omnifall.load("of-syn")` gives rows with `age_group` and `bmi_band`.
2. Take unique paths. Age groups are coded 0 toddler, 1 child, 2 teen, 3 young adult, 4 middle-aged, 5 elderly, 2,000 clips each. Keep all 2,000 elderly clips and sample a third of each other group (`groupby("age_group").sample(frac=1/3, random_state=1337)`, 667 each); save to `data/of_syn_subset.csv` (5,335 clips: train 4,274, val 522, test 539; elderly test 202 clips instead of 54 with an even split).
3. Extract with `03_extract_skeletons.py --datasets of-syn --clip-list data/of_syn_subset.csv`.
4. If extraction runs faster than expected, add more clips later; the rest stay on disk.

Both: `omnifall cite <ds>` for each dataset and paste the BibTeX into `DATA_CARD.md`; record the terms printed by `omnifall sources <ds>` beside it.

Labels and splits need no video and come down in seconds: `omnifall.load("caucafall-cs")` returns train, validation and test splits. A bare name such as `le2i` is an alias for `le2i-cs`, so always write the `-cs` suffix explicitly.

## Phase 1–2: inventory, clean and label (Dev A)

The output of this phase is a trusted list of clips with known frame rates, durations and splits, and one function that turns OmniFall segments into a label for every kept frame. No video is re-encoded: we resample to 15 fps while reading.

### Step 1 – `scripts/02_inventory.py`

1. For each dataset in the config, load labels with `omnifall.load(f"{ds}-cs")`. It returns `train`, `validation`, `test`; rename `validation` to `val`. Config names differ for the extras: of-syn for OF-Syn (random split) and of-itw for OOPS (all rows go to test). Always resolve files with each row's dataset column, never the config name.
2. Build one row per unique `(dataset, path)` with its `split`, `subject`, `cam` and the max label `end`.
3. Resolve the file: `Path(os.environ["OMNIFALL_ROOT"]) / ds / "video" / f"{path}.mp4"`.
4. Probe with OpenCV: `CAP_PROP_FPS`, `CAP_PROP_FRAME_COUNT`, width, height; read the first frame and seek to the last to prove the file decodes.
5. Flag each row (keep the row, set `flag` and `reason`):
   - `missing` – file not found
   - `unreadable` – first or last frame fails to decode
   - `short` – duration under 2 s (cannot fill one window)
   - `fps_odd` – fps outside 10–60 or reported as 0 (fall back to 25 and note it)
   - `labels_overrun` – last label `end` more than 0.5 s past the video end
6. Write `data/inventory.csv` and print a summary per dataset and split: clips, hours, subjects, number of `fall` segments.
7. Leak check, which fails the script if it trips: the set of `(dataset, subject)` pairs in `test` must not intersect `train` (ignore `subject == -1`). If a dataset has a single subject (MCFD has one actor), accept OmniFall's split, print a warning and write that into `DATA_CARD.md`.

### Step 2 – `src/fallsense/io_video.py`

Resample by timestamp, not by skipping every nth frame, so 25 fps and 30 fps sources both come out at about 15 fps with true times attached:

```python
def iter_frames(path, target_fps=15):
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step, next_t, i = 1.0 / target_fps, 0.0, 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        t = i / src_fps
        if t + 1e-6 >= next_t:
            yield int(round(t * 1000)), frame_bgr
            next_t += step
        i += 1
    cap.release()
```

Use frame index divided by fps for time, not `CAP_PROP_POS_MSEC`, which is unreliable for some codecs.

### Step 3 – `src/fallsense/labels.py`

Collapse the 10 staged classes into the 4 the model predicts. `stand_up` goes to `controlled_low` because it is a deliberate movement near the floor, not a fall:

| Our class | Id | OmniFall classes |
| --- | --- | --- |
| upright | 0 | walk (0), standing (8), other (9), and 10–15 if they ever appear |
| falling | 1 | fall (1) |
| fallen | 2 | fallen (2) |
| controlled\_low | 3 | sit\_down (3), sitting (4), lie\_down (5), lying (6), stand\_up (7) |

```python
TO4 = np.array([0, 1, 2, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
UNLABELLED = -1


def clean_segments(segs: pd.DataFrame, duration_s: float) -> pd.DataFrame:
    s = segs.copy()
    s["start"] = s["start"].clip(lower=0)
    s["end"] = s["end"].clip(upper=duration_s)
    s = s[s["end"] > s["start"]].sort_values("start")
    return s.reset_index(drop=True)


def frame_labels(segs: pd.DataFrame, t_ms: np.ndarray) -> np.ndarray:
    """16-class id per kept frame; later segments win on overlap; gaps = -1."""
    t = t_ms / 1000.0
    y = np.full(len(t), UNLABELLED, dtype=np.int8)
    for start, end, lab in segs[["start", "end", "label"]].itertuples(index=False):
        y[(t >= start) & (t < end)] = lab
    return y


def to4(y16: np.ndarray) -> np.ndarray:
    out = np.full_like(y16, UNLABELLED)
    m = y16 >= 0
    out[m] = TO4[y16[m]]
    return out
```

Cleaning rules applied inside `clean_segments` and logged as counts per dataset: negative starts clipped to 0 (the label files contain some), ends clipped to the video length, zero-length segments dropped, overlaps resolved by "later start wins", gaps left unlabelled.

### Step 4 – `tests/test_labels.py`

- A segment of 0.0–1.0 s at 15 fps labels exactly frames 0–14.
- Overlapping segments: the later one wins on the overlap.
- A gap between segments yields -1.
- `to4` maps every id 0–15 and keeps -1.
- A negative start and an end past the video are clipped, not dropped.

### Step 5 – sanity pass by eye

Pick 3 fall clips per dataset, print their segments, and play the clip with `cv2.imshow` with the current 4-class label drawn in the corner. If the label visibly lags the motion by more than about 0.3 s, stop and check the resampling before extracting anything.

## Phase 3: skeletons and features (A: extraction and QC; B: features)

Every clip becomes a 15 fps skeleton file once; everything after that reads only skeletons, which is what lets us claim training used no images. The same `pose.py` and `features.py` run offline and in the live demo, so there is no train/serve mismatch.

### A1 – `src/fallsense/pose.py` (used offline and live)

```python
import cv2, mediapipe as mp, numpy as np

V = mp.tasks.vision


class PoseExtractor:
    """One instance per clip (VIDEO mode needs strictly increasing timestamps)."""

    def __init__(self, model_path, min_det=0.5, min_presence=0.5, min_track=0.5):
        opts = V.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=V.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_det,
            min_pose_presence_confidence=min_presence,
            min_tracking_confidence=min_track,
        )
        self.lm = V.PoseLandmarker.create_from_options(opts)

    def __call__(self, frame_bgr, t_ms):
        rgb = cv2.cvtColor(
            frame_bgr, cv2.COLOR_BGR2RGB
        )  # OpenCV is BGR, MediaPipe wants RGB
        res = self.lm.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t_ms
        )
        if not res.pose_landmarks:
            return (
                np.full((33, 4), np.nan, np.float32),
                np.full((33, 3), np.nan, np.float32),
                False,
            )
        kp = np.array(
            [[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks[0]], np.float32
        )
        w = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]], np.float32)
        return kp, w, True

    def close(self):
        self.lm.close()
```

Benchmark before the full run: on 5 clips per dataset, time `pose_landmarker_lite` vs `full` and compare detection rate. Use `full` unless it runs under 10 fps per worker. For Le2i (320×240), also try upscaling 2× with `cv2.resize(..., interpolation=cv2.INTER_LINEAR)` before pose and keep whichever gives the higher detection rate during fall segments; record the choice in the config.

### A2 – `scripts/03_extract_skeletons.py`

1. Flags: `--datasets`, `--workers` (default CPU cores minus 2), `--limit N` (for smoke tests), --clip-list \<csv> (extract only the listed paths; used for the OF-Syn subset), `--overwrite`.
2. Read `data/inventory.csv`, keep rows not flagged `missing` or `unreadable`.
3. Skip clips whose .npz already exists (resumable after a crash or a lid close).
4. Run clips in a `ProcessPoolExecutor`; each worker creates its own `PoseExtractor` (MediaPipe objects cannot be pickled), loops `iter_frames(path, 15)`, stacks outputs, and saves with `np.savez_compressed` to a `.tmp` file, then renames it (no half-written files).
5. Append one line per clip to `reports/extract_log.csv`: frames, detection rate, seconds taken.
6. Expected load: about 164,000 frames across the four core datasets with all MCFD views (extras are costed in the compute budget section). At 15–30 fps per worker and 6 workers that is 20–30 minutes per laptop. If the ETA passes 2 hours, drop MCFD to 4 of its 8 cameras and log which.
7. Smoke test first: `--datasets GMDCSA24 --limit 5`, open one .npz, check shapes and that `detected.mean()` is above 0.8 for a standing-walking clip.

Run split: B extracts GMDCSA24, CAUCAFall and MCFD on B's laptop (about 120,000 frames, 15–25 min). A extracts Le2i and OCCU first (about 70,000 frames, 10–15 min), then the OF-Syn subset and OOPS (about 540,000 frames, 65–125 min) while B trains. Each zips `data/skeletons/{dataset}/` into the shared team folder (OneDrive or Google Drive; under 1 GB in total) and the other unzips it into the same path.

### A3 – `scripts/04_qc_skeletons.py`

1. For each clip compute `det_rate` (share of frames with a pose) and `det_rate_fall` (same, over frames labelled falling or fallen).
2. Identity-switch check: if the hip midpoint jumps more than 25% of the image diagonal between consecutive detected frames more than 3 times, flag `id_switch` (usually a second person in shot).
3. `qc_pass = det_rate_covered >= 0.6 and (no fall frames or det_rate_fall_covered >= 0.5) and not id_switch`, where a frame is covered if the pose is detected or the frame lies inside a `hidden_down` gap from `occlusion.find_disappearances`, and `det_rate_covered` also counts off-screen frames (Contract 2 v2). Without this, every OCCU clip fails, because fallen people behind the bed are detected in 3% of frames.
4. Apply `qc_pass` to **train and val only**. Keep every test clip and report test results both with and without failing clips; filtering test clips where pose fails during the fall would hide exactly the failures we need to show.
5. Write `data/skeletons/index.csv` (contract 2) and plots to `reports/qc/`: detection rate by dataset, and by class (expect it to drop for fallen and lying; that number goes in the pitch).
6. Render 10 random clips as skeleton-on-black MP4s in `reports/qc/samples/`. Watch them; they double as the "what the sensor sees" footage for the demo.
7. List every excluded clip and its reason in `DATA_CARD.md`.
8. Once QC passes and the shared folder is synced, the converted videos are no longer needed; the converted videos can be deleted to free space, but keep them until the model is frozen in case re-extraction is needed; both laptops stay inside their caps either way.

### B1 – `src/fallsense/features.py` (used offline and live)

One function, `make_features(kp, wh, present, ctx) -> (T, 51) float32`, used by training, evaluation and live inference. `present` and `ctx` come from Contract 3 v2 (see Occlusion handling); the four `ctx` columns are the last features. Input `kp` is (T, 13, 3): pixel x, pixel y, visibility for the 13 config keypoints (multiply normalised x by width and y by height first so angles are not distorted by the frame aspect ratio).

1. **Gap fill:** per keypoint, forward-fill NaN gaps of up to 5 frames (0.33 s); anything still missing becomes 0 with visibility 0.
2. **Scale:** torso length L = distance from shoulder midpoint (11, 12) to hip midpoint (23, 24), median over frames where all four have visibility above 0.5; fallback 0.25 × frame height; floor at 0.05 × frame height.
3. **Pose shape (26):** each keypoint's (x, y) minus that frame's hip midpoint, divided by L.
4. **Visibility (13):** raw visibility per keypoint.
5. **Derived (8), per frame:**
   - torso angle from vertical, angle of (shoulder mid − hip mid) against image up, divided by π (0 upright, 0.5 horizontal)
   - hip height = (ankle mid y − hip mid y) / L (about 2 standing, near 0 lying)
   - head height = (hip mid y − nose y) / L
   - hip vertical velocity = Δ(hip mid y) / Δt / L, from un-centred coordinates (positive = moving down)
   - hip speed = |Δ hip mid| / Δt / L
   - log bounding-box aspect = log(width / height) of visible keypoints
   - mean visibility
   - detected flag (0/1)
6. Clip every feature to \[−10, 10\] and assert no NaN leaves the function.
7. Feature z-scoring uses mean and std computed on **train windows only**, saved as `feat_stats.npz` next to the model and loaded by evaluation and live code.

`tests/test_features.py`: output shape is (30, 51); a window whose last 10 frames are not present returns finite values and carries `ctx` through; doubling all coordinates leaves shape features unchanged; a horizontally mirrored skeleton gives the same torso angle and hip height; a window with 40% NaN frames returns finite values; a synthetic "standing then lying" sequence gives torso angle rising from about 0 to about 0.5.

## Phase 4: windows, window labels and splits (Dev B)

Each training example is a 2-second, 30-frame window labelled by what happens at its end, because the live system must decide "is a fall happening now" from the last two seconds only. Splits come from OmniFall's cross-subject definitions, so no person appears in both training and test.

### Step 1 – `src/fallsense/windows.py`: the labelling rule

For a window covering frames e−29 to e, with 4-class frame labels `y4`:

1. **Drop** the window if more than 30% of its frames are unlabelled, or more than 40% have no detected pose **and no recent context** (`~present & (ctx[:, 0] >= ctx_cap_s)`); hidden-fallen windows are kept and labelled `fallen`.
2. **falling** if at least 5 frames (0.33 s) in the last 20 frames (1.33 s) are labelled falling. Falls are short, so a strict majority rule would almost never produce this class.
3. Otherwise, the most common label among the last 8 frames (about 0.5 s), ignoring unlabelled frames.

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

### Step 2 – `scripts/05_build_windows.py`

1. Read `index.csv`. Train and val use clips with `qc_pass`; test uses every clip (and a `qc_pass` column is kept so results can be cut both ways).
2. Per clip: load the .npz, take the 13 config keypoints, convert x and y to pixels with `width` and `height`, set NaN where `detected` is false. Compute `present = detected` and `ctx = occlusion.last_seen_context(...)` over the whole clip, then slice per window.
3. Build frame labels: OmniFall segments for this `(dataset, path)` → `clean_segments` → `frame_labels(segs, t_ms)` → `to4`.
4. Window end positions:
   - train and val: every 8 frames, plus every 2 frames wherever a falling frame lies within 1 s either side of the window end (more fall examples without duplicating normal activity)
   - test and staged\_test: every 3 frames everywhere (0.2 s resolution for latency)
5. Apply `window_label`; save arrays per contract 3 to `data/windows/{split}.npz` (np.savez\_compressed).
6. Write `reports/windows_summary.csv`: window counts by split × dataset × class. Expect roughly 20,000 train windows with falling at only a few percent; if falling is under 1% of train, lower the near-fall stride to 1.
7. Assertions that stop the script: no `clip_id` appears in two splits; no `(dataset, subject)` pair in train is also in test (except `subject == -1` and the single-actor MCFD case already noted); no window crosses a clip boundary; every `y` is 0–3.

### Step 3 – `src/fallsense/dataset.py`: training-time augmentation

Augment the raw keypoints, then call `make_features`, so augmentation never has to know about derived features.

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

`augment_kp`, each applied independently:

| Augmentation | Probability | Range | Simulates |
| --- | --- | --- | --- |
| Mirror left–right (flip x and swap L/R keypoint pairs) | 0.5 | – | Falls to either side |
| In-plane rotation about window hip midpoint | 0.5 | ±10° | Camera roll |
| Scale about hip midpoint | 0.5 | 0.85–1.15 | Distance to camera |
| Translation | 0.5 | ±10% of frame | Position in room |
| Coordinate jitter | 1.0 | σ = 1% of torso length | Pose noise |
| Joint dropout (set to NaN) | per joint-frame 0.05 | – | Occlusion by furniture |
| Whole-frame dropout | per frame 0.03 | – | Lost detection |
| Simulated occlusion (`occlusion.simulate_occlusion` via a `ClipBank` of train clips) | `aug_p_fall` near falls, `aug_p_other` elsewhere | `rise` or `box` | Falls behind furniture; people walking behind it |
| Speed change (resample in time, anchored at last frame) | 0.3 | 0.8–1.2× | Slower, frailer movement |

Compute `stats` (`mu`, `sd` over all train windows' features, no augmentation) once in `06_train.py` and save it with the model.

### Step 4 – `tests/test_windows.py`

- A window ending 0.5 s into a 1 s fall segment is labelled falling.
- A window ending 3 s after the fall, with the person on the floor, is labelled fallen.
- A sit-down window is labelled controlled\_low, not falling.
- A clip shorter than 30 frames yields zero windows.
- The split assertions fire on a deliberately leaked toy index.

## Phase 5: rule baseline and model training (A: baseline, B: models)

Training the skeleton model is the cheap part: a 60,000-parameter GRU on 20,000–140,000 windows takes minutes to about an hour on a laptop CPU, so run every experiment with three seeds and report the spread. The baseline exists so every model number has something honest to beat.

### A – `src/fallsense/baseline.py`

The rule from the brief: a person whose keypoint box is wider than tall and whose hips are near the floor for over 2 seconds.

1. Per frame, from `make_features`: `low = log_bbox_aspect > a and hip_height < h`.
2. FALL fires when `low` has held for 30 consecutive frames; it re-arms only after 30 frames of not-low.
3. Tune `a` in {−0.2, 0, 0.2, 0.4} and `h` in {0.3, 0.5, 0.7, 0.9} by grid search on **train** clips, maximising event F1 (event definition in the next section). Never tune on val or test.
4. Write contract-4 score files with `p_falling` = 1 on the frame the rule fires and `p_fallen` = `low`, so `08_evaluate.py` treats it exactly like a model.

### B – `src/fallsense/models.py`

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
    """Alternative: 3 dilated 1D conv blocks (dilation 1, 2, 4), kernel 5, 64 channels,
    BatchNorm + ReLU + Dropout(0.1), then concat(mean over time, last step) -> Linear."""
```

### B – `scripts/06_train.py`

1. Flags: `--model gru|tcn`, `--train-sets` (which datasets' windows to use), `--augment/--no-augment`, `--seed`, `--out models/<run-name>`.
2. Seed `random`, `numpy` and `torch`; `torch.set_num_threads(os.cpu_count() - 1)`.
3. Compute feature `mu`/`sd` over train windows (no augmentation) and save `feat_stats.npz`.
4. Loss: `CrossEntropyLoss(weight=w, label_smoothing=0.05)` with class weights `w_c = sqrt(N / (4 * n_c))`, renormalised to mean 1. Use weights, not an oversampling sampler, so the two do not double-count.
5. Optimiser: AdamW, lr 2e-3, weight decay 1e-4, batch 256, cosine decay over at most 40 epochs, gradient clipping at 1.0.
6. Selection metric on val, every epoch: average precision of falling-vs-rest (threshold-free), tie-broken by 4-class macro F1. Early stop after 8 epochs without improvement; keep the best checkpoint.
7. DataLoader: `num_workers=4, persistent_workers=True`. On Windows the whole script must sit under `if __name__ == "__main__":` or workers re-import it and crash.
8. Save to the run folder: `model.pt`, `feat_stats.npz`, a copy of the config, `train_log.csv` (epoch, train loss, val AP, val macro F1, seconds), val logits as `val_logits.npy`, and a confusion matrix PNG.
9. Print seconds per epoch after epoch 1 and an ETA. If an epoch takes over 2 minutes (likely only with OF-Syn), precompute 3 augmented copies of the train features once to `data/windows/train_aug.npz` and train from that array; it is about 10× faster.

### Experiments (one row each in `reports/experiments.csv`, 3 seeds per row)

| Id | Model | Train data | Augment | Question it answers |
| --- | --- | --- | --- | --- |
| E0 | Rule baseline | core staged | – | What does a hand rule achieve? |
| E1 | GRU | core staged | no | Does learning beat the rule? |
| E2 | GRU | core staged | yes | Does augmentation help on held-out people? |
| E3 | TCN | core staged | yes | GRU or TCN? |
| E4 | best of E2/E3 | core + OCCU | yes | Does occlusion data help? |
| E5 | best of E2/E3 | core + OCCU + OF-Syn | yes | Does synthetic diversity help, especially on older and heavier bodies? |
| E6 | best config | all except CAUCAFall | yes | How much does accuracy drop on an unseen home (CAUCAFall test)? |
| E7 | best config | same as the best of E4/E5 | yes + simulated occlusion | Does simulated occlusion help on hidden falls without more false alerts elsewhere? |

"Core staged" = GMDCSA24, CAUCAFall, Le2i, MCFD. Pick the final model on val only; open the test split once, at the end. E8 (care-home subset: OF-Syn indoor and 18+, vs all data) is deferred until a model exists; its plan, the data cuts it decides and the metrics to recompute are in `reports/data_scope_review.md`. E7's results (with and without simulated occlusion) also go into the "E7 outcome" section of `reports/occlusion_realism.md`, so the simulation can be revisited later.

### Training time (CPU, measured per epoch on the first run, then updated here)

| Train set | Approx. train windows | Per epoch | One run (early stop, ≤40 epochs) | E-row with 3 seeds |
| --- | --- | --- | --- | --- |
| Core staged | \~20,000 | 5–15 s | 3–10 min | 10–30 min |
| Core + OCCU | \~23,000 | 5–20 s | 4–12 min | 12–36 min |
| Core + OCCU + OF-Syn subset (5,335 clips) | \~59,000 | 18–48 s | 12–30 min | 36–90 min |

These are estimates for a recent 8–12-core laptop; replace them with measured numbers after the first run.

## Phase 6: calibration, CHECK thresholds and evaluation (B: calibrate, A: evaluate)

The model's raw probabilities are turned into three outputs – FALL, CHECK, silent – using two thresholds picked on validation data, and every system is scored by the same event-level script. The CHECK band is the measurable "knows when to hand over" behaviour.

### B – `src/fallsense/calibrate.py` and `scripts/07_calibrate.py`

1. **Temperature scaling.** Fit one scalar T on `val_logits.npy` by minimising negative log-likelihood (`torch.optim.LBFGS`, start T = 1.0, 50 iterations). Calibrated probabilities = `softmax(logits / T)`. Report expected calibration error before and after (15 bins).
2. **Fall score per window:** `s = p_falling`, smoothed as the max over the last 3 evaluation windows (0.6 s) to stop single-window flicker.
3. **Event logic** (shared with live code, lives in `calibrate.py`):
   - `s >= t_high` → FALL
   - `t_low <= s < t_high` → CHECK
   - `p_fallen` above 0.5 continuously for `on_floor_seconds` → ON FLOOR, whatever happened before
   - after any alert, a 30 s refractory period per person so one fall is one alert
4. **Threshold search on val only:** sweep `t_high` and `t_low` over 0.05–0.95 in steps of 0.05. Choose `t_high` as the lowest value giving at most 0.5 false FALL alerts per hour of normal activity; then choose `t_low` as the highest value that keeps event sensitivity (FALL or CHECK) at 95% or above. If no `t_low` reaches 95%, take the best achievable and say so.
5. Save `temperature`, `t_high`, `t_low` to `models/<run>/thresholds.yaml`.

### A – `src/fallsense/metrics.py` and `scripts/08_evaluate.py`

Definitions (write them in the README exactly like this):

- **Fall event:** each OmniFall `fall` segment.
- **Detected:** an alert whose time lies between the segment start and 10 s after the segment end.
- **False alert:** an alert matching no fall event.
- **Normal-activity hours:** total clip time outside \[fall start − 2 s, fall end + 10 s\].
- **Latency:** alert time minus fall segment start.

| Metric | Computed as |
| --- | --- |
| Fall sensitivity (FALL only) | detected-by-FALL events / all fall events |
| Fall sensitivity (FALL or CHECK) | detected-by-either / all fall events |
| False FALL alerts per hour | false FALL alerts / normal-activity hours |
| CHECK requests per hour | CHECK alerts / total hours |
| Coverage | share of alerts and true events decided without a CHECK |
| Median latency | median over detected events |
| Window macro F1 and confusion matrix | over evaluation windows, 4 classes |

Procedure:

1. Inputs: contract-4 score files plus `thresholds.yaml` (the baseline has none).
2. Compute all metrics for: rule baseline; model with one threshold (`t_low = t_high`); model with the CHECK band.
3. 95% intervals by bootstrap over **clips** (1,000 resamples, clips drawn with replacement), never over windows, because windows from one clip are not independent.
4. Break results down by dataset, by `qc_pass`, and for OF-Syn by `age_group` and `bmi_band`; for the staged set by lighting and occlusion tags. Add rows for model + tracker and baseline + tracker (tracker alerts from Contract 5, CHECK level), and report OCCU test and the staged hidden-fall clips on their own, with and without the tracker and with and without simulated occlusion (E7).
5. Plot the **coverage–error curve**: sweep the CHECK band width from 0 to its maximum and plot missed falls + false FALLs per hour against coverage. This is the brief's error-versus-coverage trade-off in one picture.
6. Write `reports/eval/eval_<run>_<split>.json` and a one-page `reports/eval/eval_<run>_<split>.md` with every number next to its sample size (events, clips, hours).

`tests/test_metrics.py`: a perfect detector scores sensitivity 1 and zero false alerts; an alert 11 s after a fall ends counts as false; two alerts inside one refractory window count once; bootstrap intervals contain the point estimate.

Order of use: tune and select on val → freeze model, temperature and thresholds → run test once → run staged\_test once. If anything is changed after seeing test numbers, say so in the report.

## Staged held-out test set and live script

Our own 40–60 clips, recorded in the demo space and never used for training or tuning, are the only evidence of how the model does in our room. Everyone who appears signs a consent form first, and every clip is labelled "staged" in the data and the demo.

### A – recording protocol

- **Safety:** falls only onto crash mats or stacked yoga mats, controlled and slow, never near stairs or furniture edges; a spotter stands by.
- **Cameras:** position 1 high in a corner (about 2.2 m, angled down, like a corridor mount); position 2 at about 1.2 m from the side. Same webcam model as the live demo.
- **Lighting:** normal room light and dimmed (one lamp).
- **Actors:** at least 3 team members, so results can be split by person.
- **Each clip:** 10–20 s, start standing, one action, then stay in the end state for at least 5 s.

| Action | Type | Clips (min) |
| --- | --- | --- |
| Fall forward | fall | 6 |
| Fall backward (sit then roll back) | fall | 6 |
| Fall sideways | fall | 6 |
| Slide off a chair to the floor | fall | 4 |
| Fall partly hidden behind a chair | fall | 4 |
| Fall fully hidden behind a sofa or bed (crash mat behind it) | fall | 4 |
| Walk behind the sofa, crouch out of view, reappear | upright | 3 |
| Sit on a chair, then stand | controlled | 4 |
| Sit down on the floor on purpose | controlled | 4 |
| Kneel to pick up an object | controlled | 4 |
| Lie down on the mat on purpose | controlled | 4 |
| Bend to tie a shoe | upright | 3 |
| Walk through the frame | upright | 3 |

Spread these across both camera positions and both lighting levels, and record the tags (actor, camera, lighting, occluded yes/no) in the filename: `staged/S01_cam1_dim_occ_fallfwd_01.mp4`.

### A – `scripts/record_staged.py`

OpenCV capture at 1280×720, 30 fps, writes MP4 with `cv2.VideoWriter` (`mp4v`); space bar starts and stops; the filename is built from typed tags; a red dot shows while recording.

### A – `scripts/label_tool.py`

Plays a clip at half speed; keys set the current label from that frame on: `w` walk/standing, `f` fall, `g` fallen, `s` sit\_down, `l` lie\_down, `u` stand\_up; `b` steps back 1 s, `q` saves. Output is a CSV in OmniFall format (`path, label, start, end, subject, cam, dataset="staged"`) using OmniFall class ids, so `labels.py`, `05_build_windows.py` and `08_evaluate.py` need no changes. Have a second person check every fall clip's start and end times.

Then run `03_extract_skeletons.py --datasets staged`, add the rows to `index.csv` with `split = staged_test`, and build `data/windows/staged_test.npz`.

### B – `scripts/09_live.py`

1. `cv2.VideoCapture(0)` at 640×480; one `PoseExtractor` in VIDEO mode, timestamps from `time.monotonic()` in ms (strictly increasing).
2. Keep frames at 15 fps by time, push 13-keypoint pixel coordinates into a 30-frame ring buffer.
3. Every frame: update `occlusion.LastSeenState` and `tracker.DisappearanceTracker`; keep `present` and `ctx` in the ring buffer. Every 3rd frame: `make_features(kp, wh, present, ctx)` → z-score with `feat_stats.npz` → model → `softmax(logits / T)` → the same event logic as `calibrate.py`. A tracker CHECK is shown and printed like a model CHECK, with `"source": "tracker"`.
4. Display a skeleton on a black canvas (never the camera image), the four probabilities as bars, and the current level: silent, CHECK or FALL, plus an ON FLOOR timer.
5. On every alert, print one JSON line: `{"room": "demo", "time": ..., "level": "CHECK", "score": 0.58, "source": "pose", "model": "gru-v1"}`. That line is the interface the carer app will consume later.
6. Print processing fps once a second; it must stay at 10 fps or above on the demo laptop.

## Compute budget, order of work and risks

Downloads are finished, so the remaining cost is CPU time: pose extraction takes about 15–25 min on B's laptop and 1.2–2.3 h on A's, and each training run takes minutes. Train on B's laptop while A's keeps extracting the extras.

### Why this data

The four core sets are enough to train a model and beat the baseline. OCCU, the OF-Syn subset and OOPS exist for the responsibility story: how the model does behind furniture, on older and heavier bodies, and on real accidents. The OmniFall paper's in-the-wild results (as quoted in our brief) rose from about 52% fall F1 on one staged set to about 61% on all staged sets and about 66% with synthetic video added; those figures are for a full video model, not our skeleton GRU. Experiments E4–E6 test whether each addition helps; drop anything that does not improve val.

| Dataset | Laptop | Video | Frames at 15 fps | Pose extraction | Train windows added | Why |
| --- | --- | --- | --- | --- | --- | --- |
| GMDCSA24 | B | 0.36 h | \~19,000 | 2–4 min | \~2,500 | Home rooms, webcam |
| CAUCAFall | B | 0.28 h | \~15,000 | 2–3 min | \~2,000 | Night IR, occlusion |
| MCFD (8 views) | B | 1.6 h | \~86,000 | 10–20 min | \~10,000 | High corner cameras |
| Le2i | A | 0.79 h | \~43,000 | 5–10 min | \~5,500 | Four room types |
| OCCU (2 views) | A | 0.48 h | \~26,000 | 3–6 min | \~3,000 | Falls behind furniture |
| OF-Syn subset (5,335 clips, all elderly) | A | \~7.5 h | \~400,000 | 45–95 min | \~35,000 | Older and heavier bodies |
| OOPS (test only) | A | 2.65 h | \~143,000 | 15–30 min | 0 | Real accidents |

Totals: B about 120,000 frames (15–25 min); A about 610,000 frames (1.2–2.3 h). These assume 75–150 frames/s across 6 workers on an 8–12-core laptop; replace them with the rate measured on the first clips. Training on everything (about 59,000 train windows) takes 12–30 min per run; see the Phase 5 table.

If A's extraction runs long, copy half of the OF-Syn subset videos (about 1.5 GiB) to B's laptop, which has room under its 30 GB cap, and run `03_extract_skeletons.py --datasets of-syn --clip-list` there with the other half of the list.

### Order of work (each step unblocks the next)

1. Morning checks on both laptops: every dataset verified, project venv and lockfile, environment variables, pose models, `00_check_env.py` passes.
2. **A→B:** `pose.py` and the skeleton schema frozen → B extracts GMDCSA24, CAUCAFall, MCFD; A extracts Le2i and OCCU.
3. **A→B:** `labels.py` merged → B builds the first windows from GMDCSA24 and tests `features.py` on real data.
3a. **A→B:** `occlusion.py` merged (M19, S16) → B computes `present`/`ctx` in `05_build_windows.py` and hooks simulated occlusion into `dataset.py`; A's QC uses `find_disappearances`.
4. Core skeletons QC'd and shared → B runs E1–E3 on B's laptop; A's laptop extracts the OF-Syn subset and OOPS in the background.
5. **A→B:** `08_evaluate.py` and the rule baseline (E0) ready → E1–E3 compared on val.
6. Extra skeletons QC'd and shared → E4–E6; then E7 (simulated occlusion). **A→B:** `tracker.py` merged (M20, S17) → B wires it into `09_live.py`.
7. **A→B:** staged test set recorded, labelled, extracted.
8. **B:** final model calibrated and frozen → test, staged\_test and OOPS evaluated once → report written.

### Risks

| Risk | Early sign | Fallback |
| --- | --- | --- |
| An overnight download failed | a red FAILED in the window, or `omnifall verify` reports missing files | Fallbacks in the Datasets section; rerunning `omnifall prepare` reuses archives already downloaded |
| Environment variables not picked up | `omnifall` or scripts read from `C:\Users\<name>\.cache` | Close and reopen terminals after `setx`; `OMNIFALL_ROOT` must point at an existing folder |
| OF-Syn videos are AV1 and OpenCV may not decode them | `cap.read()` returns False on the first frame | Add a PyAV branch to `iter_frames` (`av.open(path).decode(video=0)`); PyAV comes with `omnifall[video]` |
| MediaPipe loses people lying on the floor | QC plot shows detection rate for fallen far below upright | Seen on OCCU (3% fallen detection: fully hidden behind the bed). Handled by the occlusion parts 1–3 (context input, tracker, simulated occlusion); `min_presence` 0.3 only helps partial occlusion |
| Simulated occlusion unlike the real thing | realism report: simulated edge distance or descent speed far from OCCU | Tune `simulate_occlusion`; if still off, train E7 with OCCU only and say so |
| Model learns "vanished means fallen" | false FALLs when people walk out of view on val | Raise `aug_p_other`; check the context features are in the input; the tracker only raises CHECK, never FALL |
| Too few falling windows | falling under 1% in `windows_summary.csv` | Near-fall stride 1; check class weights |
| Model learns each dataset's camera rather than falls | E6 drops sharply versus E2 | More augmentation (rotation, scale), add OCCU and OF-Syn, say so in the pitch |
| Laptop sleeps mid-extraction | log stops growing | Script resumes; power plan already set to never sleep on mains |
| Live demo under 10 fps | fps print in `09_live.py` | Switch to `pose_landmarker_lite.task`; run pose on every 2nd frame |
| Labels are non-commercial | – | README states a commercial model needs its own labelled data |

## Sources

- [OmniFall dataset card (Hugging Face)](https://huggingface.co/datasets/simplexsigil2/omnifall) – components, sizes, configs, licence
- [OmniFall STRUCTURE.md](https://huggingface.co/datasets/simplexsigil2/omnifall/blob/main/STRUCTURE.md) – label and split file formats
- [OmniFall LABELS.md](https://huggingface.co/datasets/simplexsigil2/omnifall/blob/main/LABELS.md) – class definitions
- [OmniFall KNOWN\_PITFALLS.md](https://huggingface.co/datasets/simplexsigil2/omnifall/blob/main/KNOWN_PITFALLS.md) – config aliases, segment overruns
- [omnifall package on PyPI](https://pypi.org/project/omnifall/) – prepare / convert / verify commands, download sizes, env variables
- [OmniFall paper, arXiv 2505.19889](https://arxiv.org/abs/2505.19889)
- [MediaPipe Pose Landmarker, Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python) – options, VIDEO mode, outputs
- [CAUCAFall on Mendeley Data](https://data.mendeley.com/datasets/7w7fccy7ky/4)
- [GMDCSA-24 data article (Data in Brief, 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11416611/)
