import numpy as np
import numpy.typing as npt
import pytest

from fallsense import occlusion as occ

WH = (640, 480)
FPS = 15
# (dx, dy) of the 13 config keypoints as fractions of body height, y down:
# nose, shoulders, elbows, wrists, hips, knees, ankles (left, right pairs).
BODY = np.array(
    [
        (0.0, 0.0),
        (-0.10, 0.18),
        (0.10, 0.18),
        (-0.13, 0.35),
        (0.13, 0.35),
        (-0.14, 0.50),
        (0.14, 0.50),
        (-0.06, 0.52),
        (0.06, 0.52),
        (-0.06, 0.75),
        (0.06, 0.75),
        (-0.06, 1.0),
        (0.06, 1.0),
    ]
)


def _person(cx: float, top: float, height: float = 200.0) -> npt.NDArray[np.float32]:
    kp = np.ones((13, 3), np.float32)
    kp[:, 0] = cx + BODY[:, 0] * height
    kp[:, 1] = top + BODY[:, 1] * height
    return kp


def _clip(
    positions: list[tuple[float, float]], n_gone: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Visible frames at `positions` (cx, top), then `n_gone` frames with no pose."""
    kp = np.stack([_person(x, y) for x, y in positions] + [_person(0, 0)] * n_gone)
    present = np.array([True] * len(positions) + [False] * n_gone)
    kp[~present] = np.nan
    return kp, present


def _walk_off_right() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    return _clip([(300 + 11 * i, 150) for i in range(30)], n_gone=30)


def _drop_mid_frame() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    standing = [(320, 100)] * 20
    falling = [(320, 100 + 10 * i) for i in range(1, 11)]  # 150 px/s = 0.31 H/s
    return _clip(standing + falling, n_gone=30)


def test_context_is_zero_while_visible_and_counts_up_to_the_cap() -> None:
    kp, present = _clip([(320, 100)] * 10, n_gone=400)
    ctx = occ.last_seen_context(kp, present, WH, FPS, cap_s=20)
    assert (ctx[:10, 0] == 0).all()
    gone = np.arange(1, 401)
    assert np.allclose(ctx[10:, 0], np.minimum(gone / FPS, 20), atol=1e-5)
    assert ctx[-1, 0] == pytest.approx(20)
    assert np.isfinite(ctx).all()


def test_never_seen_gives_the_cap() -> None:
    kp, present = _clip([], n_gone=5)
    ctx = occ.last_seen_context(kp, present, WH, FPS, cap_s=20)
    assert (ctx[:, 0] == 20).all()
    assert np.isfinite(ctx).all()


def test_live_state_matches_whole_clip_context() -> None:
    kp, present = _drop_mid_frame()
    ctx = occ.last_seen_context(kp, present, WH, FPS)
    state = occ.LastSeenState(FPS)
    live = np.stack([state.update(kp[i], bool(present[i]), WH) for i in range(len(kp))])
    np.testing.assert_array_equal(live, ctx)


def test_walking_off_the_edge_is_left() -> None:
    kp, present = _walk_off_right()
    gaps = occ.find_disappearances(kp, present, WH, FPS)
    assert len(gaps) == 1
    assert gaps.loc[0, "kind"] == "left"
    assert gaps.loc[0, "edge_dist"] < occ.OCC["edge_margin"]


def test_dropping_mid_frame_is_hidden_down() -> None:
    kp, present = _drop_mid_frame()
    gaps = occ.find_disappearances(kp, present, WH, FPS)
    assert len(gaps) == 1
    row = gaps.loc[0]
    assert row["kind"] == "hidden_down"
    assert row["descent_speed"] > occ.OCC["descent_speed"]
    assert row["start"] == 30 and not row["returned"]


def test_short_gaps_are_ignored() -> None:
    kp, present = _clip([(320, 100)] * 20, n_gone=10)  # 0.67 s < 1 s
    assert occ.find_disappearances(kp, present, WH, FPS).empty


def test_stable_presence_fills_dropouts_and_drops_blips() -> None:
    p = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0], bool)
    want = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], bool)
    np.testing.assert_array_equal(occ.stable_presence(p, fill=2, min_run=5), want)


def test_hidden_person_flickering_back_is_one_disappearance() -> None:
    kp, present = _drop_mid_frame()
    kp[40:42], present[40:42] = kp[29], True  # detector catches them for 2 frames
    gaps = occ.find_disappearances(kp, present, WH, FPS)
    assert len(gaps) == 1
    assert gaps.loc[0, "kind"] == "hidden_down" and not gaps.loc[0, "returned"]


def _sinking(
    n: int = 60,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """A person sinking straight down from y=100 to y=340 and staying there."""
    tops = np.minimum(100 + 8 * np.arange(n), 340)
    return _clip([(320, float(t)) for t in tops], n_gone=0)


def _hidden(kp: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
    return ~(kp[..., 2] > occ.VISIBLE)  # NaN counts as hidden


def test_rise_hides_ankles_before_head_and_never_unhides() -> None:
    kp, present = _sinking()
    out, out_present = occ.simulate_occlusion(
        kp, present, WH, np.random.default_rng(3), onset=0, mode="rise"
    )
    hidden = _hidden(out)
    first = [
        int(hidden[:, k].argmax()) if hidden[:, k].any() else 10**6 for k in range(13)
    ]
    assert max(first[k] for k in occ.ANKLES) <= first[occ.NOSE]
    for k in range(13):  # once hidden in a present frame, stays hidden
        seen = hidden[out_present, k]
        if seen.any():
            assert seen[seen.argmax() :].all()
    lost = np.flatnonzero(~out_present)
    if len(lost):  # after the final loss, the person never comes back
        assert not out_present[lost[-1] :].any()


def test_frames_before_onset_are_untouched() -> None:
    kp, present = _sinking()
    for mode in ("rise", "box"):
        out, out_present = occ.simulate_occlusion(
            kp, present, WH, np.random.default_rng(0), onset=20, mode=mode
        )
        np.testing.assert_array_equal(out[:20], kp[:20])
        np.testing.assert_array_equal(out_present[:20], present[:20])


def test_hidden_keypoints_are_low_visibility_guesses_not_nan() -> None:
    kp, present = _sinking()
    out, out_present = occ.simulate_occlusion(
        kp, present, WH, np.random.default_rng(1), onset=0, mode="rise"
    )
    partly = out_present & (_hidden(out).sum(axis=1) > 0)
    assert partly.any()
    assert np.isfinite(out[partly]).all()
    assert (out[partly][..., 2] <= occ.GUESS_VIS[1] + 1e-6).any()


def test_same_seed_same_output() -> None:
    kp, present = _sinking()
    for mode in ("rise", "box"):
        a = occ.simulate_occlusion(kp, present, WH, np.random.default_rng(7), 5, mode)
        b = occ.simulate_occlusion(kp, present, WH, np.random.default_rng(7), 5, mode)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


def test_nobody_visible_after_onset_is_left_unchanged() -> None:
    kp, present = _clip([(320, 100)] * 10, n_gone=20)
    for mode in ("rise", "box"):
        out, out_present = occ.simulate_occlusion(
            kp, present, WH, np.random.default_rng(0), onset=15, mode=mode
        )
        np.testing.assert_array_equal(out, kp)
        np.testing.assert_array_equal(out_present, present)


def test_unknown_mode_rejected() -> None:
    kp, present = _sinking()
    with pytest.raises(ValueError, match="unknown occlusion mode"):
        occ.simulate_occlusion(kp, present, WH, np.random.default_rng(0), 0, "fog")
