import numpy as np
import pandas as pd
import pytest

from fallsense import metrics
from fallsense.calibrate import alerts_from_scores

FPS = 15
THRESH = {"t_high": 0.8, "t_low": 0.4}


def _scores(clip_id: int, dur_s: float, high: list[tuple[float, float, float]]):
    """Contract 4 rows every 3 frames; p_falling = value inside each (start, end)."""
    end = np.arange(29, int(dur_s * FPS), 3)
    t = end / FPS
    pf = np.zeros(len(t))
    for s, e, v in high:
        pf[(t >= s) & (t <= e)] = v
    return pd.DataFrame(
        {
            "clip_id": clip_id,
            "end_frame": end,
            "t_s": t,
            "p_upright": 1 - pf,
            "p_falling": pf,
            "p_fallen": 0.0,
            "p_controlled_low": 0.0,
        }
    )


def _evaluate(scores, clips, events, thresholds=THRESH):
    alerts = alerts_from_scores(scores, thresholds)
    clip_df, event_df = metrics.match_events(clips, events, alerts)
    return metrics.summarise(clip_df, event_df), clip_df, event_df


def _one_fall_clip(cid: int = 0, dur: float = 60.0):
    clips = pd.DataFrame({"clip_id": [cid], "duration_s": [dur]})
    events = pd.DataFrame({"clip_id": [cid], "start": [20.0], "end": [21.0]})
    return clips, events


def test_perfect_detector_finds_every_fall_with_no_false_alerts() -> None:
    clips, events = _one_fall_clip()
    res, _, _ = _evaluate(_scores(0, 60, [(20.0, 21.0, 1.0)]), clips, events)
    assert res["sensitivity_alarm"] == 1 and res["sensitivity_any"] == 1
    assert res["false_alarms"] == 0 and res["false_alarms_per_h"] == 0
    assert 0 <= res["median_latency_alarm_s"] <= 1.0
    assert res["coverage"] == 1


def test_alert_11_s_after_the_fall_ends_is_false() -> None:
    clips, events = _one_fall_clip()
    res, _, _ = _evaluate(_scores(0, 60, [(32.0, 32.5, 1.0)]), clips, events)
    assert res["sensitivity_any"] == 0
    assert res["false_alarms"] == 1


def test_alert_9_s_after_the_fall_ends_is_a_detection() -> None:
    clips, events = _one_fall_clip()
    res, _, _ = _evaluate(_scores(0, 60, [(30.0, 30.5, 1.0)]), clips, events)
    assert res["sensitivity_alarm"] == 1 and res["false_alarms"] == 0


def test_two_alerts_inside_one_refractory_window_count_once() -> None:
    clips, events = _one_fall_clip()
    scores = _scores(0, 60, [(20.0, 20.5, 1.0), (25.0, 25.5, 1.0)])
    res, clip_df, _ = _evaluate(scores, clips, events)
    assert clip_df.loc[0, "n_alarm"] == 1
    assert res["sensitivity_alarm"] == 1 and res["false_alarms"] == 0


def test_check_only_detection_counts_for_any_but_not_alarm() -> None:
    clips, events = _one_fall_clip()
    res, _, _ = _evaluate(_scores(0, 60, [(20.0, 21.0, 0.5)]), clips, events)
    assert res["sensitivity_any"] == 1 and res["sensitivity_alarm"] == 0
    assert res["checks"] == 1
    assert res["coverage"] == 0  # the one alert and the one event both went to CHECK


def test_normal_hours_exclude_the_fall_margin() -> None:
    clips, events = _one_fall_clip(dur=60.0)
    _, clip_df, _ = _evaluate(_scores(0, 60, []), clips, events)
    # 60 s minus [18 s, 31 s]
    assert clip_df.loc[0, "normal_h"] * 3600 == pytest.approx(47.0)
    assert clip_df.loc[0, "total_h"] * 3600 == pytest.approx(60.0)


def test_bootstrap_interval_contains_the_point_estimate() -> None:
    rng = np.random.default_rng(1)
    parts, clip_rows, event_rows = [], [], []
    for cid in range(40):
        found = rng.random() < 0.7
        false = rng.random() < 0.3
        high = [(20.0, 21.0, 1.0)] if found else []
        if false:
            high.append((50.0, 50.5, 1.0))
        parts.append(_scores(cid, 60, high))
        clip_rows.append((cid, 60.0))
        event_rows.append((cid, 20.0, 21.0))
    clips = pd.DataFrame(clip_rows, columns=["clip_id", "duration_s"])
    events = pd.DataFrame(event_rows, columns=["clip_id", "start", "end"])
    res, clip_df, event_df = _evaluate(pd.concat(parts), clips, events)
    ci = metrics.bootstrap(clip_df, event_df, n_boot=300)
    for name in ("sensitivity_alarm", "false_alarms_per_h", "coverage"):
        lo, hi = ci[name]
        assert lo <= res[name] <= hi, name
    assert 0 < res["sensitivity_alarm"] < 1


def test_eval_clips_matches_the_window_builder_rules() -> None:
    index = pd.DataFrame(
        {
            "dataset": ["le2i", "le2i", "le2i", "OOPS", "le2i"],
            "path": list("abcde"),
            "split": ["val", "val", "test", "test", "train"],
            "qc_pass": [True, False, False, False, True],
        }
    )
    assert metrics.eval_clips(index, "val")["clip_id"].tolist() == [0]
    assert metrics.eval_clips(index, "test")["clip_id"].tolist() == [2]
    assert metrics.eval_clips(index, "oops")["clip_id"].tolist() == [3]


def test_window_metrics_perfect_and_confused() -> None:
    y = np.array([0, 1, 2, 3, 1])
    assert metrics.window_metrics(y, y)["macro_f1"] == 1
    wrong = metrics.window_metrics(y, np.zeros_like(y))
    assert wrong["macro_f1"] < 0.5
    assert np.array(wrong["confusion"])[:, 0].sum() == len(y)
