# Laptop A session record: E8/E9, ensembles, test/OOPS, live demo (2026-10-18)

Code: commit `b0033ef` on branch `a/e8-e9` (all 9 runs trained from it, clean). Uncommitted at the end of the session:
`scripts/09_live.py` (new), the README run section, `reports/eval/*` (new reports, summary_val), `reports/experiments.csv`.

## Frozen model: `models/e9_ens_gru`

- An ensemble of e9_gru_s1, e9_gru_s2 and e9_gru_s3 (`ensemble.yaml`), with T 0.935, t_high 0.60 and t_low 0.30. ON FLOOR comes after 20 s in the demo.
- E9 recipe: E5 data (core + occu + of-syn, `--augment`), `--lr 1e-3 --weight-decay 1e-2 --low-weight 1.5 --select event --select-target 0.80`.

| | Val sens / false/h | Test sens / false/h | OOPS sens / false/h |
| --- | --- | --- | --- |
| e9_ens_gru | 81.2% / 38.6 | 82.5% / 35.7 (25-48) | 56.0% / 230 |
| e5_gru_s1 (old) | 89.6% / 113.2 | 86.4% / 112.5 (93-132) | 47.6% / 399 |
| rule baseline | 68.8% / 167.9 | 59.1% / 173.8 | 18.5% / 34.9 |

- With CHECK: 87.0% of test falls get FALL or CHECK, at 102 CHECKs/h. On test OCCU (hidden falls) it gets 50%.
- Test and OOPS were opened once, for these three systems, which were named before any number was seen.
- Decisions with all numbers: `logs/decisions.md` (E8/E9 section). Concurrency: `logs/tuning.md` (maxjobs 5, 1 thread / 4 workers; epochs ran 50-55 s at 100% CPU).

## Live demo (`scripts/09_live.py`)

- It goes PoseExtractor → LastSeenState → make_features → Scorer → AlertState, the same code as evaluation. Replaying 3 val clips (le2i, occu, of-syn) reproduces the offline scores to 1e-4.
- Cameras on Laptop A:
  - Use OpenCV's default backend (MSMF). DirectShow gave 1 fps on one camera.
  - Camera 0 is the laptop camera, 640x480 at 30 fps.
  - Camera 1 is 1280x720 at 30 fps (probably the Poly Camera Pro); it saw nobody during the test.
- Speed: about 14-15 fps steady with the full pose model, which takes 14 ms per frame. The lite model takes 11 ms.
- A seated close-up on the laptop camera hides the hips and legs (visibility 0.07) and gave one CHECK. **The camera must see the whole body.**

## Backups

- `C:\fall-data\outgoing\devA_e8-e9.zip`: E8/E9 runs, the ensembles, e5_gru_s1, and val/test/oops scores.
- Earlier backups: `devA_e4-e7.zip` and `laptopA_e4-e7_logs.zip`.
