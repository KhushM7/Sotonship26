# Decisions (Laptop A, E4-E7)

- 2026-10-17 start-up: plan A (Dev B's E1-E4 runs already unzipped; 13 model folders present). pytest 97 passed / 1 skipped; tree clean at 760f123; windows rebuilt, `reports/windows_summary.csv` unchanged, val 12,454.
- check_runs.py: Dev B's tracked eval reports are older (mtime) than the unzipped thresholds.yaml; for runs with git_commit 23288a1-dirty only, the check compares the eval JSON's t_high/t_low with thresholds.yaml instead. All 11 complete Dev B runs pass.

## Stage 1 (E5) — 2026-10-17 23:25

Ran: calibrate+evaluate e4_gru_s3, e4_tcn_s1; train+calibrate+evaluate e5_{gru,tcn}_s{1,2,3} from clean 760f123. Wall time 23:11:36-23:24:23 (~13 min). No failures, no retries. check_runs.py PASS for all 8.

Primary = fewest false FALL/h at >= 75% FALL sensitivity (mean, min-max), from logs/summary/summary_val.md:

| Row | Seeds | Primary | Sens. FALL | OF-Syn | Elderly | OCCU | Core sens. | Core false/h | Window AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e4_gru | 3 | 148.3 (133.2-173.2) | 78.7% | 78.5% | 65.3% | 47.9% | 92.1% | 215.6 | 0.68 |
| e4_tcn | 1 | 287.8 | 75.0% | 85.6% | 72.0% | 50.0% | 60.5% | 356.6 | 0.68 |
| e5_gru | 3 | 44.0 (37.3-50.6) | 85.9% | 91.1% | 90.7% | 37.5% | 93.9% | 121.6 | 0.81 |
| e5_tcn | 3 | 40.9 (37.3-45.3) | 84.5% | 88.1% | 82.7% | 35.4% | 96.5% | 118.9 | 0.81 |

1. E5 vs E4, GRU: 44.0 (37.3-50.6) vs 148.3 (133.2-173.2); ranges do not overlap -> **keep OF-Syn** (rule 3, first clause). OF-Syn sens +12.6 pts, elderly +25.4 pts also clear the second clause.
2. E5 vs E4, TCN: E4 TCN has 1 seed (287.8); treated as a point inside a range of the E4 GRU spread (±~20/h) it is far above E5 TCN's 37.3-45.3 -> **keep OF-Syn (provisional**, revisit if stage 4 runs).
3. GRU vs TCN on E5 data: 44.0 (37.3-50.6) vs 40.9 (37.3-45.3); ranges overlap -> tied on the primary number. OF-Syn 91.1% vs 88.1% (+3.0, not >10). **Tie-break on elderly sensitivity: GRU 90.7% vs TCN 82.7% -> E5 GRU wins (tie-break).**
4. **Stage 1 winner / best model so far: e5_gru (core + occu + of-syn, --augment). Seed to freeze: e5_gru_s1** — lowest primary in the row (37.3/h at t=0.70, 77.1% sens), highest calibrated FALL sensitivity (89.6%), OF-Syn 96.7%, elderly 100%. Caveat: at its calibrated t_high 0.30 it gives 113 false FALL/h (higher than s2/s3), because calibration chose the point nearest 90% sensitivity.
5. Watch: OCCU sensitivity fell with OF-Syn (GRU 47.9% -> 37.5%, TCN 50.0% -> 35.4%); OCCU val is 2 clips / 16 falls, so this is coarse. E7 (--occlude) targets exactly this.
6. Next: stage 2 = e7_gru_s{1,2,3}: E5 GRU config + --occlude.

## Stage 2 (E7) — 2026-10-17 23:45

Ran: train+calibrate+evaluate e7_gru_s{1,2,3} (E5 GRU config + --occlude, 3 threads / 4 workers) from clean 760f123. Wall time 23:31-23:43 (~12 min). No failures, no retries. check_runs.py PASS for all 3.

E7 vs its base E5 GRU (mean, min-max over 3 seeds; per-dataset rows from eval_<run>_val.json breakdowns):

| Metric | e5_gru | e7_gru |
| --- | --- | --- |
| Primary: fewest false/h at >= 75% sens | 44.0 (37.3-50.6) | 49.7 (48.0-52.0) |
| Sens. FALL (calibrated) | 85.9% (84.0-89.6) | 84.5% (81.9-87.5) |
| False FALL/h (calibrated) | 99.9 (89.3-113.2) | 78.6 (69.3-83.9) |
| OCCU sens. (16 val falls) | 37.5% (37.5-37.5) | 41.7% (37.5-50.0) |
| OCCU false/h | 91.9 (68.9-103.4) | 80.4 (68.9-103.4) |
| Core false/h (outside OCCU) | 121.6 (116.1-132.7) | 96.8 (74.6-124.4) |
| OF-Syn false/h | 96.0 (83.2-109.8) | 74.9 (66.6-83.2) |
| OF-Syn sens. | 91.1% | 87.4% |
| Elderly sens. | 90.7% (84-100) | 82.7% (76-88) |
| Window AP | 0.81 | 0.81 |

Rule 6: keep occlusion only if OCCU (or hidden-fall) sensitivity rises by more than the spread and false alerts elsewhere don't rise by more than the spread.
- OCCU: +4.2 pts on the mean, entirely from one seed (e7_gru_s2: 8/16 vs 6/16 falls); s1 and s3 equal E5 (6/16). E5's OCCU spread is 0, so "more than the spread" is degenerate at this resolution (1 fall = 6.25 pts); the E7 range (37.5-50.0) includes the E5 value. Judged **not a clear rise**. No hidden-fall breakdown exists in the eval reports; OCCU val is the proxy.
- False alerts elsewhere: did not rise (core, OF-Syn and calibrated all-val false/h all fell), so the second condition holds.
- Primary number: 49.7 vs 44.0, ranges overlap -> no change within spread. Elderly -8.0 pts (ranges overlap).
- **Decision: drop --occlude ("no clear gain"); keep the simpler E5 GRU config.** This is a judgement call on a degenerate spread: a strict reading of "mean above a zero-width range" would keep it. Flagged to Dev A in the stage report. In E7's favour for the write-up: fewer false alerts at the calibrated thresholds and on core/OF-Syn, at slightly lower sensitivity — it shifts the operating point rather than improving the trade-off at 75% sensitivity.
- **Best model so far unchanged: e5_gru, freeze e5_gru_s1.**
- Next: stage 3 = e6_gru_s{1,2,3}: E5 GRU config minus caucafall (--train-sets GMDCSA24 le2i mcfd occu of-syn --augment).

## Stage 3 (E6) — 2026-10-18 00:04

Ran: train+calibrate+evaluate e6_gru_s{1,2,3} (E5 GRU config minus caucafall: --train-sets GMDCSA24 le2i mcfd occu of-syn --augment; 3 threads / 4 workers) from clean 760f123. Base = E5 GRU (E7 was dropped). Wall time 23:56-00:03 (~7 min). No failures, no retries. check_runs.py PASS for all 3. Train 51,973 windows (vs 53,429); selection on 12,052 val windows (caucafall val excluded).

Rule 5 — CAUCAFall rows of the per-dataset breakdown (val: 8 clips, 4 falls, 0.0135 normal hours ≈ 49 s):

| Run | Sens. FALL | False alarms (count) | False/h | CHECK coverage |
| --- | --- | --- | --- | --- |
| e5_gru_s1 / s2 / s3 | 4/4, 4/4, 4/4 | 3, 3, 3 | 222, 222, 222 | 1.00, 0.72, 0.86 |
| e6_gru_s1 / s2 / s3 | 4/4, 4/4, 4/4 | 2, 1, 3 | 148, 74, 222 | 0.80, 0.67, 0.86 |

- **E6 result: no measurable drop on the unseen home.** Every E6 seed still catches all 4 CAUCAFall val falls, and false alarms there are 1-3 per seed vs 3 for E5 (one alarm = 74/h on 49 s of normal video, so these rates are noise). CAUCAFall val is too small to detect a drop; record it as "no drop detected, val too small to show one", not as "generalises".
- All of val, E6 vs E5 GRU: primary 40.4 (36.0-49.3) vs 44.0 (37.3-50.6), ranges overlap; Sens. FALL 84.5% vs 85.9%; elderly 84.0% vs 90.7% (ranges overlap); window AP 0.81 vs 0.81. Leaving CAUCAFall out of training changes nothing measurable elsewhere either.
- E6 is a measurement row, not a candidate. **Best model so far unchanged: e5_gru, freeze e5_gru_s1.**
- Next: stage 4 (optional) = e4_tcn_s2, e4_tcn_s3 (--model tcn --train-sets GMDCSA24 caucafall le2i mcfd occu --augment), to put the E4 -> E5 TCN comparison on 3 seeds.

## Stage 4 (complete E4 TCN) — 2026-10-18 00:10

Ran: train+calibrate+evaluate e4_tcn_s2, e4_tcn_s3 (--model tcn --train-sets GMDCSA24 caucafall le2i mcfd occu --augment; 3 threads / 4 workers) from clean 760f123. Wall time 00:05-00:09 (~4 min; 21,331 train windows, early stop ~epoch 10). No failures, no retries. check_runs.py PASS for both. Note: e4_tcn_s1 is Dev B's (23288a1-dirty, AnanthHN_Laptop); s2/s3 are Laptop A (760f123) — mixed provenance within the row.

- E4 TCN now 3 seeds: primary 275.8 (263.8-287.8), Sens. FALL 75.2%, OF-Syn 82.6%, elderly 72.0%, OCCU 43.8%, core false/h 329.0.
- **Revisit of the provisional stage 1 TCN decision: E5 TCN 40.9 (37.3-45.3) vs E4 TCN 275.8 (263.8-287.8); ranges do not overlap -> keep OF-Syn for TCN, now final** (no longer provisional).
- E4 GRU vs TCN (for the write-up): GRU 148.3 (133.2-173.2) vs TCN 275.8 (263.8-287.8), no overlap -> GRU better on core+occu.
- **Final config: e5_gru (core + occu + of-syn, --augment, no --occlude). Freeze e5_gru_s1.** All stages ran; next is the section 6 write-up.

## Write-up — 2026-10-18 00:25

- Tracked summaries written after the last training run: `reports/eval/summary_val.{md,csv}` (`--target 0.75`, baseline + all 27 runs E1-E7), `reports/experiments.csv` (E1-E3 rows identical to HEAD; E4 GRU calibrated 2 -> 3; E4 TCN 3 seeds; E5, E6, E7 added).
- e4_gru_s1/s2: tracked reports were older than thresholds.yaml by mtime only (unzip artifact). Re-ran 08_evaluate on both as a check: output byte-identical to the tracked files (git shows no change).
- `reports/occlusion_realism.md` "E7 outcome" filled in; plan docs ticked; Step 0 row A and the master plan's "Training time" table filled in from measured numbers.
- Final: check_runs.py PASS for all 27 runs (logs/check_runs_final.txt); pytest 97 passed / 1 skipped.

## E8/E9 + ensembles — 2026-10-18 02:35 (commit b0033ef, branch a/e8-e9)

New recipe (lr 1e-3, weight decay 1e-2, epoch chosen by fewest false FALL/h at >= 80% sens on val); E9 adds 1.5x controlled_low class weight. Calibration targets now 80% FALL / 88% FALL-or-CHECK (reachable), giving a real CHECK band (t_low < t_high). All 9 runs + 5 ensembles pass check_runs.py. maxjobs 6 -> 5 at 02:11 (CPU 100%).

Best operating point (fewest false/h at >= 75% sens, val): e5_gru 44.0 (37.3-50.6), e8_gru 35.5 (29.3-38.6), e9_gru 34.2 (33.3-36.0), e9_tcn 37.3 (33.3-42.6); ensembles e5_ens_gru 37.3, e8_ens_gru 30.6, e9_ens_gru 25.3, e9_ens_tcn 25.3, e9_ens_all 28.0.
Calibrated (sens / false/h / core false/h): e9_ens_gru 81.2% / 38.6 / 24.9; e9_ens_all 81.2% / 36.0 / 33.2; e9_ens_tcn 80.6% / 40.0 / 49.8; e5_ens_gru 80.6% / 57.3 / 66.3.

- Recipe (E8 vs E5 GRU): 35.5 vs 44.0, ranges overlap only at the edge (E8 max 38.6 vs E5 min 37.3) -> modest, consistent gain.
- controlled_low weight (E9 vs E8 GRU): 34.2 vs 35.5, overlap -> no clear single-model gain; ensemble 25.3 vs 30.6 and core false/h 24.9 vs 33.2 favour E9. Keep it.
- Ensembling: every ensemble beats its seeds' mean (e9 gru 34.2 -> 25.3).
- Core-only best-point column is 0.00 for most rows: core val has 38 falls and little normal time, so it can't separate runs.
- Caveat: E8/E9 choose epochs on the same val event metric they are judged on, so their val numbers are more optimistic than E5's. Test decides whether the gain is real.
- Trade-off: at the new 80% operating point, elderly OF-Syn sensitivity is 76% (e5_gru_s1 at its 0.30 threshold: 100%, at 113 false/h).
- **Freeze e9_ens_gru** (e9_gru_s1..3): ties best operating point, lowest core false/h, 3 GRUs = cheap live. Thresholds from models/e9_ens_gru/thresholds.yaml.
- Test/OOPS opened once, for e9_ens_gru, e5_gru_s1 (previous freeze, reference) and the rule baseline, named before any test number was seen.
