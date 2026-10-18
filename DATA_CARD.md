# Data card – fallsense

What data trained and evaluated the frozen model (`models/e9_ens_gru`), where it came from, under which terms, and what was left out. Numbers come from `data/skeletons/index.csv` (Contract 2), `reports/windows_summary.csv` and `reports/qc/excluded.csv`; regenerate them with `scripts/04_qc_skeletons.py` and `scripts/05_build_windows.py`.

## Summary

- **Source:** seven public datasets, all through the [OmniFall](https://huggingface.co/datasets/simplexsigil2/omnifall) benchmark (labels, cross-subject splits, conversion tool `omnifall`).
- **What was used:** MediaPipe skeletons only (13 of 33 keypoints, 15 fps). No model was trained on pixels; the live demo never shows the camera image.
- **Size:** 6,802 clips, 14.5 h of video; 6,475 clips pass quality control.
- **Splits:** OmniFall cross-subject (`-cs`): no person appears in two of train, val and test. OOPS is test-only; MCFD (one actor) is train-only.
- **Licence of the result:** the labels are non-commercial (CC BY-NC-SA 4.0), so the trained model is for research and demonstration only. **A commercial model needs its own labelled data.**
- **No own recordings.** The planned staged test set (our own room, consenting actors) was not recorded; it was dropped for time. There is therefore no `staged_test` split.

## Datasets

| Dataset | Licence (`omnifall sources`) | What it is | Clips | Hours | Role |
| --- | --- | --- | --- | --- | --- |
| [GMDCSA24](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos) | MIT (repository licence) | 4 actors, 3 homes, laptop webcam | 160 | 0.36 | train / val / test |
| [CAUCAFall](https://data.mendeley.com/datasets/7w7fccy7ky/4) | CC BY 4.0 | 10 subjects, home-like room, night infrared and occlusion | 97 | 0.27 | train / val / test |
| [MCFD](https://www.iro.umontreal.ca/~labimage/Dataset/) | see homepage (citation requested; not machine-readable) | 24 scenarios × 8 high corner cameras, one actor | 192 | 2.42 | train only |
| [Le2i](https://search-data.ubfc.fr/imvia/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html) | CC BY-NC-SA 3.0 | home, office, coffee room, lecture room; 320×240 | 190 | 0.85 | train / val / test |
| [OCCU](https://doi.org/10.5281/zenodo.15494102) | CC BY 4.0 | long bedroom recordings, 2 views, falls hidden behind the bed | 10 | 0.48 | train / val / test |
| [OF-Syn](https://huggingface.co/datasets/simplexsigil2/omnifall) (subset) | CC BY-NC-SA 4.0 (README) / CC BY-NC 4.0 (card metadata) | AI-generated videos with age group, BMI, camera and environment metadata | 5,335 of 12,000 | 7.51 | train / val / test |
| [OOPS-Fall](https://oops.cs.columbia.edu/data/) | CC BY-NC-SA 4.0, non-commercial research and education | real accidents from online "fail" videos | 818 | 2.65 | out-of-domain test only |
| **Total** | | | **6,802** | **14.52** | |

**OmniFall labels and splits:** CC BY-NC-SA 4.0. The OmniFall Hub card metadata says CC BY-NC 4.0 while its README says CC BY-NC-SA 4.0 (the ShareAlike clause differs); we follow the stricter reading. The original videos belong to their owners and were downloaded from the original sources by `omnifall prepare`.

**CAUCAFall** has 100 videos in the release, but OmniFall labels only 97; the other 3 are not used.

**Not used:** UP-Fall (slow frame-to-video rebuild), CMDFall (access by email only), EDF (same lab and setup as OCCU without the occlusion). Long untrimmed daily-living sets (Toyota Smarthome Untrimmed, Charades) were considered for measuring false alerts over long periods and skipped for time and licence reasons.

## OF-Syn subset

OF-Syn has 12,000 clips in six age groups (toddler, child, teen, young adult, middle-aged, elderly; 2,000 each). We kept **all 2,000 elderly clips** – the population this is for – plus a third of every other group: 5,335 clips, listed in `configs/of_syn_subset.csv` (`path, age_group, bmi_band, split, activity`). This gives 202 elderly test clips instead of about 54 with an even sample. Results are broken down by age group and BMI band in every evaluation report.

## Clips per split

Clips in `index.csv` (QC-passing in brackets) and hours of video.

| Dataset | Train | Val | Test |
| --- | --- | --- | --- |
| GMDCSA24 | 75 (72), 0.15 h | 48 (47), 0.13 h | 37 (36), 0.08 h |
| CAUCAFall | 69 (64), 0.18 h | 9 (8), 0.03 h | 19 (14), 0.06 h |
| MCFD | 192 (118), 2.42 h | – | – |
| Le2i | 133 (126), 0.62 h | 19 (15), 0.06 h | 38 (33), 0.17 h |
| OCCU | 6 (6), 0.29 h | 2 (2), 0.08 h | 2 (2), 0.10 h |
| OF-Syn | 4,274 (4,261), 6.02 h | 522 (517), 0.73 h | 539 (535), 0.76 h |
| OOPS | – | – | 818 (619), 2.65 h |
| **Total** | **4,749 (4,647)** | **600 (589)** | **635 + 818 OOPS** |

Training windows (`data/windows/*.npz`, 30 frames = 2 s each): train 53,429, val 12,454, test 14,673, OOPS 38,872. Train windows end every 8 frames (every 2 within 1 s of a fall); val, test and OOPS windows end every 3 frames, like the live system.

## What was excluded and why

**Quality control** (`scripts/04_qc_skeletons.py`): a clip passes if the person is found or explainably out of sight (hidden after going down, or off-screen at a frame border) in at least 60% of frames, in at least 50% of its fall frames, with at most 3 identity jumps. Failing clips are **removed from train and val only** (113 clips, each listed with its reason in `reports/qc/excluded.csv`); every test and OOPS clip is kept, and reports show results with and without them.

| Dataset | Failing clips (all splits) | Removed from train/val | Reason |
| --- | --- | --- | --- |
| MCFD | 74 of 192 | 74 | low detection; cameras 4 and 5 find the person in only 34–44% of fall frames |
| OF-Syn | 22 of 5,335 | 18 | low detection |
| Le2i | 16 of 190 | 11 | low detection |
| CAUCAFall | 11 of 97 | 6 | low detection |
| GMDCSA24 | 5 of 160 | 4 | low detection |
| OCCU | 0 of 10 | 0 | – (hidden falls count as covered) |
| OOPS | 199 of 818 | – (test only, kept) | low detection; 10 identity switches to another person |
| **Total** | **327 of 6,802** | **113** | |

**Windows** are dropped if more than 30% of their frames are unlabelled, or more than 40% have no pose and no recent "last seen" context.

**Label cleaning** (`src/fallsense/labels.py`): negative start times clipped to 0, end times clipped to the video length, zero-length segments dropped, overlaps resolved by "later segment wins", gaps left unlabelled. OmniFall's 16 classes are mapped to four (`upright`, `falling`, `fallen`, `controlled_low`); kneeling, squatting, crawling and jumping count as `upright`.

## Known issues in the data

- **Actors and synthetic people.** Almost all falls are staged by actors or generated. Real falls of frail older people are often slower, partial and against furniture. OOPS (real accidents, mostly young people doing sport or stunts) is reported separately for this reason: the frozen model catches 56% of OOPS falls against 82.5% on the in-domain test set.
- **Single person.** Pose runs with `num_poses: 1`. At least ~10% of OOPS clips and ~3% of OF-Syn clips have two or more people; MediaPipe sometimes tracks a bystander. The system is scoped to one person per room.
- **Domain.** 52% of OF-Syn clips are outdoors and 37% show under-18s. A care-home-only subset was considered (`reports/data_scope_review.md`) and not trained.
- **Little normal activity.** Val has 0.75 h and test 0.90 h of normal activity, so false-alert rates per hour have wide intervals.
- **Short clips.** Public clips are short, so ON FLOOR is evaluated after 5 s instead of the demo's 20 s.

## Storage and deletion

- Videos were converted locally by `omnifall prepare`; only skeleton `.npz` files (Contract 1) were shared between the two laptops.
- Nothing under `data/` (videos, skeletons, windows) is committed. Converted videos are kept on the team laptops until the hackathon is over and then deleted; downloaded archives likewise.
- No video or images of the team were recorded for training or evaluation. The live demo processes webcam frames in memory and can save only skeletons, scores and alerts (`09_live.py --record`).

## Citations

Cite OmniFall and each component dataset (`omnifall cite <dataset>`):

```bibtex
@misc{omnifall,
  title={OmniFall: From Staged Through Synthetic to Wild, A Unified Multi-Domain Dataset for Robust Fall Detection},
  author={David Schneider and Zdravko Marinov and Rafael Baur and Zeyun Zhong and Rodi Düger and Rainer Stiefelhagen},
  year={2025}, eprint={2505.19889}, archivePrefix={arXiv}, primaryClass={cs.CV},
  url={https://arxiv.org/abs/2505.19889}
}
@article{omnifall_gmdcsa,
  title={GMDCSA-24: A dataset for human fall detection in videos},
  author={Alam, Ekram and Sufian, Abu and Dutta, Paramartha and Leo, Marco and Hameed, Ibrahim A},
  journal={Data in Brief}, volume={57}, pages={110892}, year={2024}, publisher={Elsevier}
}
@article{omnifall_cauca,
  title={Dataset CAUCAFall},
  author={Eraso, Jose Camilo and Mu{\~n}oz, Elena and Mu{\~n}oz, Mariela and Pinto, Jesus},
  journal={Mendeley Data}, volume={4}, year={2022}
}
@article{omnifall_mcfd,
  title={Multiple cameras fall dataset},
  author={Auvinet, Edouard and Rougier, Caroline and Meunier, Jean and St-Arnaud, Alain and Rousseau, Jacqueline},
  journal={DIRO-Universit{\'e} de Montr{\'e}al, Tech. Rep}, volume={1350}, pages={24}, year={2010}
}
@article{omnifall_le2i,
  title={Optimized spatio-temporal descriptors for real-time fall detection: comparison of support vector machine and Adaboost-based classification},
  author={Charfi, Imen and Miteran, Johel and Dubois, Julien and Atri, Mohamed and Tourki, Rached},
  journal={Journal of Electronic Imaging}, volume={22}, number={4}, pages={041106--041106}, year={2013}
}
@inproceedings{omnifall_edf_occu,
  title={Evaluating depth-based computer vision methods for fall detection under occlusions},
  author={Zhang, Zhong and Conly, Christopher and Athitsos, Vassilis},
  booktitle={International symposium on visual computing}, pages={196--207}, year={2014}, organization={Springer}
}
@inproceedings{omnifall_oops,
  title={Oops! predicting unintentional action in video},
  author={Epstein, Dave and Chen, Boyuan and Vondrick, Carl},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={919--929}, year={2020}
}
```

OF-Syn is part of the OmniFall release and is covered by the OmniFall citation. MediaPipe pose models: Apache 2.0.
