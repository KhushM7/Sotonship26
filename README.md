# fallsense

Skeleton-based fall detection (hackathon project). The full README follows at the end
of the project (setup, pipeline order, data card, AI-assistance disclosure).

## Evaluation metrics

Computed by `scripts/08_evaluate.py` (`src/fallsense/metrics.py`) for every system
the same way: models, the rule baseline and the disappearance tracker.

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

FALL and ON FLOOR alerts both count as alarms. 95% intervals are a bootstrap over
clips (1,000 resamples with replacement), never over windows.
