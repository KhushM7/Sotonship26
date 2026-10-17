# Dev A Plan – Data In (Fall Detection)

Oct 17, 2026 · @K

## How to use this doc

Dev A owns **data in**: the repo, getting video into skeletons, labels, QC, the rule baseline, the evaluator and the staged test set. Dev B follows a matching doc ([Dev B Plan – Model Out](https://claude.ai/code/artifact/6fdc9db6-9f2b-4db2-8f7d-1f734fcd97fe)); the full reference is the [master build plan](https://claude.ai/code/artifact/a8b9bcc7-5227-445b-ba58-900b7e0cc8bd).

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

Compute times in this doc are estimates; replace them with measured numbers as you go.

## Step 0 – Morning download checks

Your laptop holds Le2i, OCCU, OF-Syn and OOPS in `C:\fall-data\cache\videos\<dataset>\video\`. Confirm every set is complete before any code is written.

- [ ] In any new PowerShell window, set the session folders:
  - `$env:OMNIFALL_CACHE_DIR = "C:\fall-data\cache"`
  - `$env:OMNIFALL_DOWNLOAD_DIR = "C:\fall-data\downloads"`
  - `Remove-Item Env:OMNIFALL_ROOT -ErrorAction SilentlyContinue`
- [ ] OOPS: answer `y` at the licence prompt if still waiting; wait for `OOPS DONE - 818 of 818 videos`.
- [ ] `omnifall verify le2i`, `omnifall verify occu`, `omnifall verify of-syn`, `omnifall verify OOPS` – each reports no missing files.
- [ ] Video counts: `(Get-ChildItem C:\fall-data\cache\videos\<dataset>\video -Recurse -Filter *.mp4).Count` – expect Le2i 190, OOPS 818, OF-Syn 12,000; note OCCU's count.
- [ ] Any failure: Le2i – rerun the resumable curl block into `C:\fall-data\downloads\le2i\FallDataset.zip`, then `omnifall prepare le2i --workers 4`. Anything else – rerun `omnifall prepare <dataset> --workers 4` (it reuses archives on disk).
- [ ] Free space once everything verifies: `Remove-Item C:\fall-data\downloads\* -Recurse -Force` (removes the 45 GB OOPS archive and the Le2i zip).
- [ ] Record each dataset's licence from `omnifall sources <dataset>` and BibTeX from `omnifall cite <dataset>` in a scratch note; it goes into `DATA_CARD.md` in Step 7.
- [ ] **SYNC S0 (both)** – read your four verify results out loud; Dev B reads theirs (GMDCSA24, CAUCAFall, MCFD). Both tick only when all seven datasets verify.

## Step 1 – Create the repo and environment

You create the repo; Dev B clones it and adds the config and environment-check files. Use Python 3.11 only (MediaPipe does not support 3.13+).

### 1a. Repo skeleton

- [ ] On GitHub create a **private** repo `fallsense`; Settings → Collaborators → add Dev B.
- [ ] `git clone <repo-url> C:\code\fallsense` and `cd C:\code\fallsense`.
- [ ] Create the layout:

```powershell
New-Item -ItemType Directory -Force configs, src\fallsense, scripts, tests, models\pose, models\gru-v1, reports | Out-Null
New-Item -ItemType File -Force src\fallsense\__init__.py, README.md, DATA_CARD.md | Out-Null
```

- [ ] `.gitignore`:

```
data/
*.mp4
*.avi
*.npz
.venv/
models/pose/*.task
__pycache__/
.env.ps1
```

- [ ] `requirements.txt`:

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

- [ ] `.env.ps1.example` (each dev copies it to `.env.ps1`):

```powershell
$env:FALL_DATA = "C:\fall-data"
$env:OMNIFALL_CACHE_DIR = "C:\fall-data\cache"
$env:OMNIFALL_DOWNLOAD_DIR = "C:\fall-data\downloads"
$env:OMNIFALL_ROOT = "C:\fall-data\cache\videos"
```

- [ ] `pyproject.toml` with a minimal `[project] name = "fallsense"` and `[tool.setuptools.packages.find] where = ["src"]`, so `pip install -e .` makes `import fallsense` work.
- [ ] Commit and push `straight to main (M1, the only direct push); from here on every change goes through a branch and pull request`.
- [ ] **SYNC S1 (A → B)** – tell Dev B the repo URL. Dev B clones it and pushes a branch `b/hello`; you see it on GitHub. Tick when confirmed out loud.

### 1b. Environment and lockfile

- [ ] From the repo root:
  - `uv venv --python 3.11 .venv`
  - `.venv\Scripts\activate`
  - `uv pip install -r requirements.txt`
  - `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`
  - `uv pip install -e .`
- [ ] `uv pip freeze > requirements.lock`, commit it on branch a/lockfile and merge by pull request (M2), then tell Dev B. Dev B installs with `uv pip sync requirements.lock`.
- [ ] Link data into the repo: `New-Item -ItemType Junction -Path data -Target C:\fall-data`.
- [ ] Save environment variables once, then close and reopen all terminals:
  - `setx FALL_DATA "C:\fall-data"`
  - `setx OMNIFALL_CACHE_DIR "C:\fall-data\cache"`
  - `setx OMNIFALL_DOWNLOAD_DIR "C:\fall-data\downloads"`
  - `setx OMNIFALL_ROOT "C:\fall-data\cache\videos"`
- [ ] Pose models (use `curl.exe`, not `curl`):
  - `curl.exe -L -o models\pose\pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task`
  - same again with `full` replaced by `lite` in both places. If a link 404s, copy the current one from the Models section of the [Pose Landmarker overview](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker).
- [ ] Pull Dev B's `config.py`, `configs/default.yaml` and `scripts/00_check_env.py` once Dev B has merged them to main (M3); review that pull request.
- [ ] **SYNC S2 (A ↔ B)** – both run `python scripts\00_check_env.py`; every line prints OK on both laptops. Read out `mediapipe` and `torch` versions; they must match. Tick only when both pass.

### Git habits (both)

Branch per task (`a/labels`, `a/pose`, `b/features`), small PRs, the other dev reviews before merge, `ruff format` and `pytest -q` before every push, nothing under `data/` ever committed.

### Branches and merges to `main` (identical in both docs)

Each dev works on their own branches and merges small pieces to `main` as soon as another piece needs them. Merge points are numbered **M1–M17** and appear again at the step where they happen.

Rules:

- `main` always runs: the only direct push is M1; every other change goes in through a pull request.
- The other dev reviews and approves every PR; the author has run `ruff format` and `pytest -q` first.
- Use squash and merge, then delete the branch.
- Right after any merge, both devs run `git checkout main` and `git pull`, then `git rebase main` on whatever branch they are on.
- Merge small and early; a branch should hold one piece, not a day's work.
- Only the file's owner edits it (see the repo layout). A dev who needs a change in the other's file asks for it or opens a small PR that the owner approves; `configs/default.yaml` belongs to Dev B, so Dev A's `le2i_upscale` change goes through B.
- Never commit anything under `data/`; skeletons, windows and models travel through the shared folder.
- Verbal check for every merge: the author says "M-number merged"; the other dev pulls `main`, runs `pytest -q`, and says "main is green".

| Merge | Who | Branch | What goes in | Merge when | Needed by |
| --- | --- | --- | --- | --- | --- |
| M1 | A | `main` (direct) | Repo skeleton, `.gitignore`, `requirements.txt`, `pyproject.toml`, `.env.ps1.example` | Before S1 | S1 |
| M2 | A | `a/lockfile` | `requirements.lock` | Your own install works | S2 |
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

## Step 2 – Data contracts (identical in both docs)

These formats let both devs code in parallel. Change one only if both agree, and bump `schema_version` in `configs/default.yaml`.

### Repo layout and owners

```
fallsense/
  configs/default.yaml           B
  src/fallsense/
    config.py                    B
    labels.py  io_video.py  pose.py  baseline.py  metrics.py      A
    features.py  windows.py  dataset.py  models.py  calibrate.py  B
  scripts/
    00_check_env.py  05_build_windows.py  06_train.py  07_calibrate.py  09_live.py     B
    02_inventory.py  03_extract_skeletons.py  04_qc_skeletons.py  08_evaluate.py
    record_staged.py  label_tool.py                                               A
  tests/   test_labels.py test_metrics.py (A)   test_features.py test_windows.py test_model.py (B)
  data/ -> junction to C:\fall-data (gitignored)
```

### Config values both sides rely on

`fps: 15`, `window: 30` frames (2 s), `stride_train: 8`, `stride_train_near_fall: 2`, `stride_eval: 3`, `keypoints: [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]` (nose, shoulders, elbows, wrists, hips, knees, ankles), `classes: [upright, falling, fallen, controlled_low]`, `datasets: [GMDCSA24, caucafall, le2i, mcfd, occu, of-syn]`, `test_only_datasets: [OOPS]`, `seed: 1337`, `on_floor_seconds: 20`.

### Contract 1 – skeleton file (you write, everyone reads)

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

### Contract 2 – skeleton index (you write)

`data/skeletons/index.csv`: `dataset, path, subject, cam, split, n_frames, duration_s, width, height, det_rate, det_rate_fall, qc_pass, qc_reason`; `split` is `train`, `val`, `test` or `staged_test`.

### Contract 3 – window arrays (Dev B writes)

`data/windows/{split}.npz`: `kp` float32 (N, 30, 13, 3) pixel x, y, visibility; `wh` int16 (N, 2); `y` int64 (N,) 0–3; `clip_id` int32 (row in `index.csv`); `end_frame` int32.

### Contract 4 – frame scores (Dev B writes for models, you write for the baseline; your evaluator reads)

`reports/scores/{model}/{split}.parquet`: `clip_id, end_frame, t_s, p_upright, p_falling, p_fallen, p_controlled_low`.

- [ ] **SYNC S3 (both)** – read this section together; both say "agreed" before writing any code.

## Step 3 – Inventory, video reader, pose wrapper and extraction script

Dev B cannot extract their three datasets until these four files are merged, so they come first. Both laptops run the same scripts on their own videos.

### 3a. `scripts/02_inventory.py`

- [ ] Flag `--datasets` (default: whatever is on this laptop under `OMNIFALL_ROOT`).
- [ ] Load labels with `omnifall.load(config)`: `"<ds>-cs"` for GMDCSA24, caucafall, le2i, mcfd, occu; `"of-syn"` for OF-Syn; `"of-itw"` for OOPS (all OOPS rows go to `test`). Rename split `validation` to `val`.
- [ ] One row per unique `(dataset, path)` with `split`, `subject`, `cam`, max label `end`. Resolve the file with each row's `dataset` column, never the config name: `Path(os.environ["OMNIFALL_ROOT"]) / dataset / "video" / f"{path}.mp4"`.
- [ ] Probe with OpenCV: fps, frame count, width, height; decode the first and last frame.
- [ ] Flag (keep the row, fill `flag`, `reason`): `missing`, `unreadable`, `short` (< 2 s), `fps_odd` (outside 10–60 or 0; fall back to 25), `labels_overrun` (last label end > video end + 0.5 s).
- [ ] Write `data/inventory.csv` (local to each laptop) and print clips, hours, subjects and fall-segment counts per dataset and split.
- [ ] Leak check that fails the script: no `(dataset, subject)` pair in both `train` and `test` (ignore `subject == -1`). MCFD has one actor: accept OmniFall's split, print a warning, note it for `DATA_CARD.md`.

### 3b. `src/fallsense/io_video.py`

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

- [ ] Use frame index / fps for time, not `CAP_PROP_POS_MSEC`.
- [ ] Test on one OF-Syn clip: if the first `cap.read()` returns False (OF-Syn is AV1), add a PyAV branch (`av.open(path)`, `container.decode(video=0)`, `frame.to_ndarray(format="bgr24")`); PyAV comes with `omnifall[video]`.

### 3c. `src/fallsense/pose.py`

```python
import cv2, mediapipe as mp, numpy as np
V = mp.tasks.vision

class PoseExtractor:
    """One instance per clip (VIDEO mode needs strictly increasing timestamps)."""
    def __init__(self, model_path, min_det=0.5, min_presence=0.5, min_track=0.5):
        opts = V.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=V.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=min_det,
            min_pose_presence_confidence=min_presence,
            min_tracking_confidence=min_track)
        self.lm = V.PoseLandmarker.create_from_options(opts)

    def __call__(self, frame_bgr, t_ms):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)   # MediaPipe wants RGB
        res = self.lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t_ms)
        if not res.pose_landmarks:
            return np.full((33, 4), np.nan, np.float32), np.full((33, 3), np.nan, np.float32), False
        kp = np.array([[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks[0]], np.float32)
        w = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]], np.float32)
        return kp, w, True

    def close(self):
        self.lm.close()
```

- [ ] Benchmark on 5 clips each of Le2i and OCCU: `pose_landmarker_lite` vs `full`, frames/s per worker and detection rate. Use `full` unless it runs under 10 fps per worker.
- [ ] Le2i is 320×240: also try upscaling 2× (`cv2.resize(..., interpolation=cv2.INTER_LINEAR)`) before pose; keep whichever gives the higher detection rate during fall segments, and write the choice into the config (tell Dev B).

### 3d. `scripts/03_extract_skeletons.py`

- [ ] Flags: `--datasets`, `--workers` (default CPU cores − 2), `--limit N`, `--clip-list <csv>` (only these paths; used for the OF-Syn subset), `--overwrite`.
- [ ] Input: `data/inventory.csv` rows not flagged `missing` or `unreadable`.
- [ ] Resumable: skip clips whose .npz exists.
- [ ] `ProcessPoolExecutor`; each worker builds its own `PoseExtractor` per clip, loops `iter_frames(path, 15)`, stacks outputs, saves with `np.savez_compressed` to a `.tmp` then renames.
- [ ] Append to `reports/extract_log.csv`: dataset, path, frames, detection rate, seconds.
- [ ] Print an ETA after the first 10 clips.
- [ ] Smoke test yourself: `--datasets le2i --limit 5`; open one .npz, check shapes against Contract 1 and `detected.mean()` above 0.8 on a walking clip.
- [ ] Merge 3a–3d `by pull request from branch a/io-pose (M4); Dev B reviews and you both confirm main is green before S4`.
- [ ] **SYNC S4 (A → B)** – tell Dev B the four files are merged and which pose model (and Le2i upscale setting) to use. Dev B pulls, runs `02_inventory.py` and `03_extract_skeletons.py --datasets GMDCSA24 --limit 5`, and reads out the array shapes and detection rate. Tick when both agree it works.

## Step 4 – Labels, extraction, QC and the extra datasets

You write the label mapping, extract your own datasets, collect Dev B's skeletons, and QC everything into one `index.csv`. Dev B's windowing waits on S5 and S7.

### 4a. `src/fallsense/labels.py`

OmniFall classes: 0 walk, 1 fall, 2 fallen, 3 sit\_down, 4 sitting, 5 lie\_down, 6 lying, 7 stand\_up, 8 standing, 9 other, 10–15 extended (kneel, squat, crawl, jump; OOPS and OF-Syn only).

| Our class | Id | OmniFall classes |
| --- | --- | --- |
| upright | 0 | walk, standing, other, 10–15 |
| falling | 1 | fall |
| fallen | 2 | fallen |
| controlled\_low | 3 | sit\_down, sitting, lie\_down, lying, stand\_up |

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

def segments_for(dataset: str, path: str) -> pd.DataFrame:
    """All OmniFall segments for one clip (cache the loaded configs)."""
```

- [ ] Log per dataset how many segments had negative starts clipped, ends clipped, zero length dropped, overlaps resolved.
- [ ] `tests/test_labels.py`: 0.0–1.0 s at 15 fps labels frames 0–14; later segment wins on overlap; gaps give -1; `to4` maps 0–15 and keeps -1; negative start and over-long end are clipped, not dropped.
- [ ] Sanity pass: play 3 fall clips per dataset with `cv2.imshow` and the 4-class label drawn in the corner. If the label lags the motion by more than about 0.3 s, fix the resampling before extracting.
- [ ] Merge `by pull request from branch a/labels (M5) once the tests pass and Dev B has reviewed; main is green before S5`.
- [ ] **SYNC S5 (A → B)** – Dev B pulls, runs `pytest tests/test_labels.py`, and you both look at the frame labels of one fall clip. Tick when Dev B confirms.

### 4b. Extract your core and occlusion sets

- [ ] `python scripts\02_inventory.py --datasets le2i occu`
- [ ] `python scripts\03_extract_skeletons.py --datasets le2i occu` (estimate: 70,000 frames, 10–15 min). Keep the laptop plugged in; the script resumes if interrupted.

### 4c. Collect Dev B's skeletons

- [ ] Shared team folder (OneDrive or Google Drive): `fallsense-data/skeletons/`. Zip each dataset folder from `data\skeletons\` and upload yours too.
- [ ] **SYNC S6 (B → A)** – Dev B says GMDCSA24, CAUCAFall and MCFD are uploaded with their `extract_log.csv`. You unzip into `data\skeletons\` and check the .npz count per dataset matches their log (GMDCSA24 160, CAUCAFall 100, MCFD per their log). Tick when the counts match out loud.

### 4d. `scripts/04_qc_skeletons.py`

- [ ] Read every .npz under `data\skeletons\`; take `split` from the OmniFall configs (labels only, no video needed), so it works for skeletons from either laptop.
- [ ] Per clip: `det_rate`, and `det_rate_fall` over frames labelled falling or fallen.
- [ ] Identity switch: hip midpoint jumps more than 25% of the image diagonal between consecutive detected frames more than 3 times → `id_switch`.
- [ ] `qc_pass = det_rate >= 0.6 and (no fall frames or det_rate_fall >= 0.5) and not id_switch`.
- [ ] Apply `qc_pass` to train and val only; keep every test clip (results get reported both with and without failing clips).
- [ ] Write `data\skeletons\index.csv` (Contract 2), `reports\qc\det_rate_by_dataset.png`, `reports\qc\det_rate_by_class.png`.
- [ ] Render 10 random clips as skeleton-on-black MP4s in `reports\qc\samples\` and watch them; they are the "what the sensor sees" demo footage.
- [ ] List every excluded clip and its reason for `DATA_CARD.md`. Merge 04\_qc\_skeletons.py by pull request from branch a/qc (M8) before S7.
- [ ] Upload `index.csv` and `reports\qc\` to the shared folder.
- [ ] **SYNC S7 (A → B)** – Dev B downloads the Le2i skeletons and `index.csv`, and reads out index row count and `qc_pass` counts per dataset; you read yours; they match. Tick when confirmed.

### 4e. OF-Syn subset and OOPS (runs while Dev B trains)

- [ ] Subset: `omnifall.load("of-syn")`, unique paths, sample 4,000 stratified by `age_group` (`groupby("age_group").sample(frac=..., random_state=1337)`; keep all elderly clips if that group is small). Save `data\of_syn_subset.csv` and commit a copy to `configs/` on branch a/extras and merge by pull request (M12) before S11.
- [ ] `python scripts\02_inventory.py --datasets of-syn OOPS`
- [ ] `python scripts\03_extract_skeletons.py --datasets of-syn --clip-list data\of_syn_subset.csv` (estimate 35–70 min).
- [ ] `python scripts\03_extract_skeletons.py --datasets OOPS` (estimate 15–30 min).
- [ ] If this runs long, copy half of the subset's OF-Syn videos (about 1.5 GiB) to Dev B with a second clip list; Dev B has room under 30 GB.
- [ ] Rerun `04_qc_skeletons.py`; upload the new zips, `index.csv` and QC plots.
- [ ] **SYNC S11 (A → B)** – Dev B downloads them, rebuilds windows, and reads out window counts for OCCU, OF-Syn and OOPS. Tick when confirmed.

Disk: once QC passes, converted videos can be deleted, but keep them until S14 in case re-extraction is needed.

## Step 5 – Rule baseline and evaluator

The evaluator scores every system the same way, and the baseline gives every model number something honest to beat. Both depend on code from Dev B: `features.py` (S8) and the event logic in `calibrate.py` (S9).

- [ ] **SYNC S8 (B → A)** – Dev B says `features.py` is merged. You pull and run `pytest tests/test_features.py`; it passes. Tick when confirmed.

### 5a. `src/fallsense/metrics.py`

Definitions (copy them into the README word for word):

- **Fall event:** each OmniFall `fall` segment.
- **Detected:** an alert between the segment start and 10 s after the segment end.
- **False alert:** an alert matching no fall event.
- **Normal-activity hours:** clip time outside \[fall start − 2 s, fall end + 10 s\].
- **Latency:** alert time minus fall segment start.

| Metric | Computed as |
| --- | --- |
| Fall sensitivity (FALL only) | events detected by FALL / all fall events |
| Fall sensitivity (FALL or CHECK) | events detected by either / all fall events |
| False FALL alerts per hour | false FALL alerts / normal-activity hours |
| CHECK requests per hour | CHECK alerts / total hours |
| Coverage | share of alerts and true events decided without a CHECK |
| Median latency | over detected events |
| Window macro F1, confusion matrix | over evaluation windows, 4 classes |

- [ ] Bootstrap 95% intervals over **clips** (1,000 resamples with replacement), never over windows.
- [ ] Alerts come from Dev B's `alerts_from_scores(scores, thresholds)`; do not reimplement the event logic.
- [ ] **SYNC S9 (B → A)** – Dev B says `alerts_from_scores` and the contract-4 score writer are merged. You run `tests/test_metrics.py` against them; it passes. Tick when confirmed.
- [ ] `tests/test_metrics.py`: a perfect detector gives sensitivity 1 and zero false alerts; an alert 11 s after a fall ends counts as false; two alerts inside one 30 s refractory window count once; bootstrap intervals contain the point estimate.

### 5b. `src/fallsense/baseline.py`

The rule from the brief: keypoint box wider than tall and hips near the floor for over 2 s.

- [ ] Per frame, from `make_features` on the whole clip: `low = log_bbox_aspect > a and hip_height < h`.
- [ ] FALL fires when `low` has held for 30 consecutive frames; re-arms after 30 frames of not-low.
- [ ] Grid search `a` in {−0.2, 0, 0.2, 0.4} and `h` in {0.3, 0.5, 0.7, 0.9} on **train** clips only, maximising event F1.
- [ ] Write contract-4 score files to `reports\scores\baseline\{split}.parquet` every 3 frames: `p_falling` = 1 on the frame the rule fires, `p_fallen` = `low`, others 0. Save `t_high = t_low = 0.5` in `reports\scores\baseline\thresholds.yaml`.

### 5c. `scripts/08_evaluate.py`

- [ ] Inputs: `--model <run>` (score folder) and `--split`; reads contract-4 scores, `index.csv`, OmniFall labels, and the run's `thresholds.yaml`.
- [ ] Report three rows: rule baseline; model with one threshold (`t_low = t_high`); model with the CHECK band.
- [ ] Break down by dataset, by `qc_pass`, by OF-Syn `age_group` and `bmi_band`, and for `staged_test` by lighting and occlusion tags from file names.
- [ ] Plot the coverage–error curve: sweep CHECK band width from 0 to max; missed falls + false FALLs per hour against coverage.
- [ ] Write `reports\eval_<run>_<split>.json` and a one-page `.md` with every number next to its sample size (events, clips, hours).
- [ ] Run the baseline (E0) on val, then rebase on main and merge by pull request from branch a/eval (M11); Dev B needs it on main for S10.
- [ ] **SYNC S10 (A → B)** – tell Dev B the evaluator and E0 val results are merged. Dev B runs `08_evaluate.py` on their E1 val scores; you both read out sensitivity and false FALL alerts per hour for E0 and E1. Tick when the numbers look sensible to both.

Rule for test data: tune and select on val only. Test, staged\_test and OOPS are evaluated once, at S14. If anything changes after seeing test numbers, the report says so.

## Step 6 – Staged test set

Your own 40–60 clips, recorded in the demo space and never used for training or tuning, are the only evidence of how the model does in your room. You run the session; Dev B and at least one other team member act and spot.

### 6a. Tools

- [ ] `scripts/record_staged.py`: OpenCV capture at 1280×720, 30 fps; `cv2.VideoWriter` with `mp4v`; space bar starts and stops; filename built from typed tags; red dot while recording. Saves to `C:\fall-data\staged\`.
- [ ] `scripts/label_tool.py`: plays a clip at half speed; keys set the label from that frame on: `w` walk/standing, `f` fall, `g` fallen, `s` sit\_down, `l` lie\_down, `u` stand\_up; `b` back 1 s; `q` saves. Output: CSV in OmniFall format (`path, label, start, end, subject, cam, dataset="staged"`) with OmniFall class ids, so `labels.py` and the evaluator need no changes.
- [ ] Consent form: one page; name, date, agrees to being recorded performing staged falls for a hackathon demo, clips labelled as staged, deleted after the event unless agreed otherwise. Print or use a shared form. Merge record\_staged.py and label\_tool.py (with the staged support described in 6c) by pull request from branch a/staged-tools (M13) before the recording session.

### 6b. Recording protocol

- Safety: falls only onto crash mats or stacked yoga mats, slow and controlled, never near stairs or furniture edges; a spotter stands by.
- Cameras: position 1 high in a corner (about 2.2 m, angled down); position 2 at about 1.2 m from the side. Same webcam model as the live demo.
- Lighting: normal room light and dimmed (one lamp).
- Actors: at least 3 people, so results can be split by person.
- Each clip: 10–20 s, start standing, one action, hold the end state at least 5 s.
- File name: `S01_cam1_dim_occ_fallfwd_01.mp4` (actor, camera, lighting, occluded yes/no, action, take).

| Action | Type | Clips (min) |
| --- | --- | --- |
| Fall forward | fall | 6 |
| Fall backward (sit then roll back) | fall | 6 |
| Fall sideways | fall | 6 |
| Slide off a chair to the floor | fall | 4 |
| Fall partly hidden behind a chair | fall | 4 |
| Sit on a chair, then stand | controlled | 4 |
| Sit down on the floor on purpose | controlled | 4 |
| Kneel to pick up an object | controlled | 4 |
| Lie down on the mat on purpose | controlled | 4 |
| Bend to tie a shoe | upright | 3 |
| Walk through the frame | upright | 3 |

Spread actions across both camera positions and both lighting levels.

- [ ] Every actor signs the consent form before recording.
- [ ] Record the session.
- [ ] **SYNC S12 (both)** – walk the table above together and tick off coverage (actions × cameras × lighting × actors). Record any missing clips before packing up. Tick when both agree coverage is complete.

### 6c. Label, extract, hand over

- [ ] Label every clip with `label_tool.py`; save `C:\fall-data\staged\labels.csv`.
- [ ] Dev B independently checks the start and end of every fall segment and corrects any that are off by more than 0.2 s.
- [ ] Teach 02\_inventory.py and labels.segments\_for to read C:\\fall-data\\staged\\labels.csv when the dataset is staged (it is not an OmniFall config), then add staged clips to the inventory and extract: `python scripts\02_inventory.py --datasets staged` and `python scripts\03_extract_skeletons.py --datasets staged`.
- [ ] Rerun `04_qc_skeletons.py`; staged rows get `split = staged_test`. Upload the staged skeleton zip, labels CSV and `index.csv`.
- [ ] **SYNC S13 (A → B)** – Dev B confirms the fall times were double-checked, downloads the files, builds `staged_test.npz`, and reads out clip and window counts. Tick when confirmed.

## Step 7 – Final evaluation, documentation and done check

- [ ] **SYNC S14 (B → A)** – Dev B says the model is frozen and uploads `models\<run>\model.pt`, `feat_stats.npz`, `thresholds.yaml` and score files for `test`, `staged_test` and OOPS. You confirm the files are there and the run name matches. Tick when confirmed.
- [ ] Run `08_evaluate.py` once each on `test`, `staged_test` and OOPS for the frozen model and the baseline.
- [ ] Review the reports with Dev B: headline numbers with sample sizes and intervals, coverage–error curve, subgroup table (OF-Syn age and BMI), detection rate by class from QC.

### DATA\_CARD.md

- [ ] One row per dataset: source link, licence (from `omnifall sources`), BibTeX (from `omnifall cite`), clips used per split, hours, what was excluded and why. Licences recorded so far: Le2i CC BY-NC-SA 3.0, OOPS CC BY-NC-SA 4.0, OmniFall labels CC BY-NC-SA 4.0 (the card metadata says CC BY-NC 4.0; note the inconsistency), CAUCAFall CC BY 4.0, GMDCSA24 MIT.
- [ ] MCFD single-actor split note; OF-Syn subset method and the committed clip list; staged clips labelled "staged" with consent recorded.
- [ ] Statement: training used skeletons only; converted videos deleted after the event; the labels are non-commercial, so a commercial model needs its own labelled data.

### README.md (your parts; Dev B adds the model section)

- [ ] Setup steps from Step 1; data pipeline order (inventory → extract → QC → windows → train → calibrate → evaluate); metric definitions from Step 5; links to `DATA_CARD.md` and the eval reports; AI-assistance disclosure. Merge DATA\_CARD.md and these README parts by pull request from branch a/docs (M16) before S15; after S15 passes, tag main as v1-hackathon (M17).

### Your risks to watch

| Risk | Early sign | Fallback |
| --- | --- | --- |
| OF-Syn is AV1 and OpenCV cannot decode it | first `cap.read()` returns False | PyAV branch in `iter_frames` |
| MediaPipe loses people on the floor | QC by-class plot: fallen far below upright | Re-extract with `min_presence` 0.3; report as a known limit |
| Laptop sleeps mid-extraction | `extract_log.csv` stops growing | Script resumes; power plan already set to never sleep on mains |
| Env vars not picked up | scripts read `C:\Users\<name>\.cache` | Reopen terminals after `setx`; `OMNIFALL_ROOT` must exist |
| Extraction slower than estimated | ETA print after 10 clips | Switch to `pose_landmarker_lite`; share OF-Syn work with Dev B |

### Definition of done

- [ ] `data\skeletons\` has one .npz per clip for every dataset used, all through QC
- [ ] `data\windows\` has train, val, test, staged\_test and OOPS arrays from person-disjoint splits (Dev B)
- [ ] Baseline and GRU scored by the same `08_evaluate.py` on the same windows and events
- [ ] Model, temperature and thresholds saved in `models\<run>\` (Dev B)
- [ ] `09_live.py` shows skeleton and alert level on a webcam at 10 fps or more (Dev B)
- [ ] `DATA_CARD.md` complete
- [ ] **SYNC S15 (both)** – Dev B demos `09_live.py` (skeleton on black, never the camera image, 10+ fps); walk this checklist together and tick each item out loud.
