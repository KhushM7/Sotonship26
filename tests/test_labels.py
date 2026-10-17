from collections import Counter

import numpy as np
import pandas as pd

from fallsense.labels import UNLABELLED, clean_segments, frame_labels, to4

FPS = 15


def _segs(*rows):
    return pd.DataFrame(rows, columns=["start", "end", "label"])


def _t_ms(n):
    return np.round(np.arange(n) * 1000 / FPS).astype(np.int64)


def test_one_second_segment_labels_frames_0_to_14():
    y = frame_labels(_segs((0.0, 1.0, 1)), _t_ms(30))
    assert (y[:15] == 1).all()
    assert (y[15:] == UNLABELLED).all()


def test_later_segment_wins_on_overlap():
    segs = clean_segments(_segs((0.0, 2.0, 0), (1.0, 3.0, 1)), duration_s=5.0)
    y = frame_labels(segs, _t_ms(45))
    t = _t_ms(45) / 1000
    assert (y[t < 1.0] == 0).all()
    assert (y[(t >= 1.0) & (t < 3.0)] == 1).all()


def test_gap_between_segments_is_unlabelled():
    y = frame_labels(_segs((0.0, 1.0, 0), (2.0, 3.0, 2)), _t_ms(45))
    t = _t_ms(45) / 1000
    assert (y[(t >= 1.0) & (t < 2.0)] == UNLABELLED).all()
    assert (y[(t >= 2.0) & (t < 3.0)] == 2).all()


def test_to4_maps_every_id_and_keeps_unlabelled():
    y16 = np.array([-1, *range(16)], dtype=np.int8)
    expected = [-1, 0, 1, 2, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0]
    assert to4(y16).tolist() == expected
    assert to4(y16).dtype == y16.dtype


def test_negative_start_and_long_end_are_clipped_not_dropped():
    counts = Counter()
    out = clean_segments(
        _segs((-0.4, 1.0, 1), (1.0, 9.0, 2)), duration_s=5.0, counts=counts
    )
    assert len(out) == 2
    assert out.loc[0, "start"] == 0.0
    assert out.loc[1, "end"] == 5.0
    assert counts["neg_start_clipped"] == 1
    assert counts["end_clipped"] == 1
    assert counts["zero_length_dropped"] == 0


def test_segment_entirely_past_the_end_is_dropped_and_counted():
    counts = Counter()
    out = clean_segments(
        _segs((0.0, 2.0, 0), (6.0, 7.0, 2)), duration_s=5.0, counts=counts
    )
    assert len(out) == 1
    assert counts["zero_length_dropped"] == 1


def test_overlaps_are_counted():
    counts = Counter()
    clean_segments(
        _segs((0.0, 2.0, 0), (1.5, 3.0, 1), (3.0, 4.0, 2)),
        duration_s=5.0,
        counts=counts,
    )
    assert counts["overlaps"] == 1


def test_touching_segments_with_float32_rounding_are_not_overlaps():
    counts = Counter()
    segs = _segs(
        (0.0, 1.5500600337982178, 0), (1.55006, 3.0, 1)
    )  # as stored in float32
    clean_segments(segs, duration_s=5.0, counts=counts)
    assert counts["overlaps"] == 0
