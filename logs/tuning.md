# Concurrency tuning log (Laptop A)

| Time | Change | Reason / measurement |
| --- | --- | --- |
| 2026-10-17 start | maxjobs 6, threads 2, workers 3, 60 s gap | Section 3.3 starting configuration. Power mode set to Best performance (was default overlay). On AC. OneDrive running (not checked whether sync is paused). |
| 2026-10-17 start | e5_gru_s3 launched with threads 1 / workers 4 | Section 3.4 workers-vs-threads trial; compare its s/epoch with e5_gru_s1/s2 |
| 2026-10-17 23:21 | no maxjobs change | Stage 1 never reached 6 concurrent: E5 runs take ~5 min (early stop at epoch 11-14, 14-30 s/epoch) and starts are 60 s apart, so peak was 5 training runs. Solo (e5_tcn_s3 at the tail): 13.9 s/epoch = 3,844 train windows/s. Peak (5 runs, ~23:17-23:19): 17-26 s/epoch each, total ≈ 12,600 windows/s (≈3.3x solo). CPU sample at the tail (1 run training + 2 evals): 43%, 40 GB free. No thrashing, no swapping; RAM is not a limit. |
| 2026-10-17 23:21 | workers-vs-threads trial: threads 1 / workers 4 is faster | e5_gru_s3 (1/4) ran 16-18 s/epoch while e5_gru_s2 (2/3) ran 21-31 s/epoch over the same minutes (same data, same model). Augmentation workers are the bottleneck, not torch math. |
| 2026-10-17 stage 2 start | threads 3, workers 4, maxjobs 6 (3 runs) | Section 3.4: stages 2-4 have 3 runs, so 3/4 to fill the cores. Stage 1 showed 1/4 beat 2/3 at 5 concurrent; re-measure at epoch 3. |
| 2026-10-17 23:38 | no change (all 3 stage-2 runs already running) | E7 (--occlude, 3 threads / 4 workers, 3 concurrent): CPU 86% avg, 32 GB free, total ≈ 5,700 windows/s. s/epoch 15-20 at low contention (≈1.2x E5's 14-16), rising to 30-43 for e7_gru_s1/s2 later (contention/thermal). No swapping. |
| 2026-10-17 stage 3 start | threads 3, workers 4, 3 runs | unchanged from stage 2 (86% CPU, no thrashing) |
| 2026-10-18 stage 4 start | threads 3, workers 4, 2 runs | unchanged |
| 2026-10-18 E8/E9 start | maxjobs 6, threads 1, workers 4, 60 s gap | 9 runs; stage 1 of E4-E7 showed 1/4 beats 2/3 at 5-6 concurrent |
| 2026-10-18 02:11 | maxjobs 6 -> 5 | CPU 100% with 6 training runs + a live-replay check (full pose model); E8/E9 epochs 50-55 s vs 17-26 s in E5 (lower lr also means more epochs). 19 GB RAM free, no swapping. Keep 1/4. |
