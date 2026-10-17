# Data scope review – revisit after the first trained model (E8)

**Status (17 Oct): deferred.** We train on all QC-passing data first and decide on cuts
with a model in hand. Nothing below has been applied to the data yet. Revisit once E1–E5
have val results, before the final model is chosen.

## Experiment E8 – care-home subset vs everything

Train the best config so far twice: once on all train data, once on the care-home subset
(drop OF-Syn outdoor and OF-Syn under 18; OOPS is test-only already). Compare both on
the same in-domain val/test rows: Le2i, GMDCSA24, CAUCAFall, OCCU and OF-Syn indoor
elderly. Report falling-vs-rest AP, 4-class macro F1, recall at the operating point and
false alerts per hour, with sample sizes. Keep the cut only if the subset wins on the
in-domain rows. Also report OOPS separately as out-of-domain for both runs.

Run names, results and the keep/drop decision go in **E8 outcome** below and in
`reports/experiments.csv`.

## What was measured (17 Oct, on 6,802 clips; scratch scripts, not in the repo)

| Possible cut | Clips | % of all clips | Train falls lost (of 1,299) |
|---|---|---|---|
| Skeleton stays upright through a labelled fall (wrong person tracked) | ~30 falls in ~28 clips | ~0.4% | ~13 |
| Multi-person clips (OOPS ≥79, OF-Syn ~3%; MediaPipe misses small people, so a minimum) | ~250 (est.) | ~3.7% | ~40 (est.) |
| Already excluded by QC (train/val) | 113 | 1.7% | – |
| OOPS (test only) | 818 | 12.0% | 0 |
| OF-Syn outdoor | 2,756 | 40.5% | 425 |
| OF-Syn under 18 | 2,001 | 29.4% | 282 |
| OF-Syn outdoor or under 18 | 3,716 | 54.6% | 557 |
| OF-Syn except indoor elderly | 4,333 | 63.7% | 668 |
| Care-home only (OOPS + OF-Syn outdoor or under 18) | 4,534 | 66.7% | 557 (keeps 742) |

Data-validity checks tried:

- **"Did the skeleton go down during the fall?"** (hips drop ≥ 0.25 body heights or torso
  tilts ≥ 30°): flags 22% of falls, but the OF-Syn ones checked on video were real falls
  (onto hands and knees, back into sitting, clips that start mid-fall). Do not use.
- **"Skeleton stays upright and still through the whole fall"** (tilt < 25°, full height,
  no hip drop, fall + 1 s): flags 30 falls (17 OOPS, 11 OF-Syn, 2 MCFD). All 3 OOPS cases
  checked were the wrong person (bystander, man beside a ladder, trumpet player); 1 of 2
  OF-Syn cases was a false flag. Good candidate for a `label_conflict` QC reason
  (train/val only; report test with and without).
- **Anatomy check** (left/right limb lengths, leg/torso ratio, visible limbs only):
  implausible frames 3.5–4.3% on Le2i, CAUCAFall and OF-Syn, 7.5% MCFD, 10.2% OOPS.
  Signal but noisy: use per frame (treat as no pose), not to drop clips.

## Known pose problems seen in the sample videos (not fixed)

- Multiple people: MediaPipe (`num_poses: 1`) locks onto the wrong person, mostly in OOPS.
- Nobody detected in some still scenes (no face visible, small or lying person).
- A human skeleton fitted to a dog.
- Skeletons merging when two people are close.
- Kneeling, squatting and crawling (OmniFall 10–14) map to `upright`; they may belong
  in `controlled_low` (about 2% of OF-Syn, plus some OOPS).

Options and difficulty were discussed on 17 Oct: plausibility filter (easy),
multi-person flag via `num_poses: 3` (easy), single-person scope in `DATA_CARD.md`
(trivial), detection settings for still scenes (easy, needs re-extraction),
multi-person tracking (medium–hard), separate person detector (hard).

## Metrics to recompute when revisiting

Rerun on the then-current data and compare with the table above:

1. QC pass counts per dataset (`scripts/04_qc_skeletons.py`).
2. Fall agreement per dataset: agrees / stayed upright / hidden / unverified.
3. Implausible-frame share per dataset.
4. Multi-person share per dataset (Dev B runs it for GMDCSA24, CAUCAFall, MCFD).
5. Model errors by these flags: are false negatives and false alerts concentrated in
   multi-person, label-conflict, outdoor or under-18 clips? If not, the cuts are not
   worth the lost data.

## E8 outcome

*To be filled in after E8: run names, in-domain and OOPS results for both runs, and the
decision on each cut above.*
