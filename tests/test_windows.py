from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from fallsense import windows


def _toy(n: int = 5) -> dict[str, npt.NDArray[Any]]:
    return {
        "kp": np.random.rand(n, 30, 13, 3),
        "wh": np.tile([640, 480], (n, 1)),
        "y": np.arange(n) % 4,
        "clip_id": np.arange(n),
        "end_frame": np.arange(n) * 3 + 29,
        "present": np.random.rand(n, 30) > 0.2,
        "ctx": np.random.rand(n, 30, 4),
    }


@pytest.fixture
def tmp_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        windows, "windows_path", lambda split: tmp_path / f"{split}.npz"
    )


def test_round_trip_dtypes_and_shapes(tmp_windows: None) -> None:
    windows.save_windows("val", **_toy())
    d = windows.load_windows("val")
    for key, (dtype, tail) in windows.SCHEMA.items():
        assert d[key].dtype == dtype
        assert d[key].shape == (5, *tail)


def test_empty_split_is_valid(tmp_windows: None) -> None:
    windows.save_windows("oops", **_toy(0))
    assert len(windows.load_windows("oops")["y"]) == 0


def test_bad_label_rejected(tmp_windows: None) -> None:
    t = _toy()
    t["y"][0] = 4
    with pytest.raises(ValueError, match="y must be"):
        windows.save_windows("train", **t)


def test_bad_shape_rejected(tmp_windows: None) -> None:
    t = _toy()
    t["kp"] = t["kp"][:, :20]
    with pytest.raises(ValueError, match="kp: shape"):
        windows.save_windows("train", **t)


def test_v2_keys_round_trip(tmp_windows: None) -> None:
    t = _toy()
    windows.save_windows("val", **t)
    d = windows.load_windows("val")
    assert d["present"].dtype == np.bool_ and d["present"].shape == (5, 30)
    assert d["ctx"].dtype == np.float32 and d["ctx"].shape == (5, 30, 4)
    np.testing.assert_array_equal(d["present"], t["present"])
    np.testing.assert_allclose(d["ctx"], t["ctx"], rtol=1e-6)


def test_missing_v2_key_rejected() -> None:
    t: dict[str, npt.NDArray[Any]] = {
        k: np.asarray(v, dtype=windows.SCHEMA[k][0]) for k, v in _toy().items()
    }
    del t["ctx"]
    with pytest.raises(ValueError, match="missing keys"):
        windows.validate_windows(t)


def test_bad_ctx_shape_rejected(tmp_windows: None) -> None:
    t = _toy()
    t["ctx"] = t["ctx"][..., :3]
    with pytest.raises(ValueError, match="ctx: shape"):
        windows.save_windows("train", **t)


def test_bad_present_shape_rejected(tmp_windows: None) -> None:
    t = _toy()
    t["present"] = t["present"][:, :20]
    with pytest.raises(ValueError, match="present: shape"):
        windows.save_windows("train", **t)


def test_nan_ctx_rejected(tmp_windows: None) -> None:
    t = _toy()
    t["ctx"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="ctx must be finite"):
        windows.save_windows("train", **t)


def test_config_is_schema_v2() -> None:
    from fallsense.config import load_config

    cfg = load_config()
    assert cfg["schema_version"] == 2
    occ = cfg["occlusion"]
    assert set(occ) == {
        "ctx_cap_s",
        "edge_margin",
        "descent_speed",
        "hidden_check_s",
        "min_visible_kp",
        "lookback_frames",
        "aug_p_fall",
        "aug_p_other",
    }
    # lookback covers one window plus the context cap
    assert occ["lookback_frames"] == cfg["window"] + occ["ctx_cap_s"] * cfg["fps"]


# ---------------------------------------------------------------- windowing rules (M7)

FPS = 15
CAP = 20.0


def _labels(*runs: tuple[int, int]) -> np.ndarray:
    """Frame labels from (class, n_frames) runs."""
    return np.concatenate([np.full(n, c, dtype=np.int8) for c, n in runs])


def _all_visible(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.ones(n, bool), np.zeros((n, 4), np.float32)


def _label_at(
    y4: np.ndarray, end: int, present: np.ndarray | None = None
) -> int | None:
    present = np.ones(len(y4), bool) if present is None else present
    s = end - 29
    return windows.window_label(y4[s : end + 1], present[s : end + 1])


def test_half_second_into_one_second_fall_is_falling() -> None:
    y4 = _labels((0, 60), (1, 15), (2, 60))  # fall from frame 60 to 75
    assert _label_at(y4, 60 + 7) == 1


def test_three_seconds_after_fall_is_fallen() -> None:
    y4 = _labels((0, 60), (1, 15), (2, 90))
    assert _label_at(y4, 75 + 45) == 2


def test_sit_down_is_controlled_low() -> None:
    y4 = _labels((0, 40), (3, 40))  # sit_down / sitting map to 3
    assert _label_at(y4, 70) == 3


def test_mostly_unlabelled_window_dropped() -> None:
    y4 = _labels((-1, 12), (0, 18))
    assert _label_at(y4, 29) is None


def test_hidden_after_fall_kept_but_long_gone_dropped() -> None:
    y4 = _labels((0, 60), (1, 15), (2, 90))
    present = np.arange(len(y4)) < 75  # hidden behind the bed after the fall
    ctx = np.zeros((len(y4), 4), np.float32)
    ctx[:, 0] = np.clip((np.arange(len(y4)) - 74) / FPS, 0, CAP)
    recent = present | (ctx[:, 0] < CAP)
    assert _label_at(y4, 100, recent) == 2
    never_seen = np.zeros(len(y4), bool)
    assert _label_at(y4, 100, never_seen) is None


def test_short_clip_gives_no_windows() -> None:
    y4 = _labels((0, 29))
    assert len(windows.window_ends(29, y4, train=True)) == 0
    present, ctx = _all_visible(29)
    out = windows.clip_windows(
        np.zeros((29, 13, 3)), present, ctx, y4, np.zeros(0, int), CAP
    )
    assert out["kp"].shape == (0, 30, 13, 3) and len(out["y"]) == 0


def test_window_ends_strides() -> None:
    y4 = _labels((0, 100), (1, 15), (2, 100))
    ev = windows.window_ends(len(y4), y4, train=False)
    assert ev[0] == 29 and (np.diff(ev) == 3).all()
    tr = windows.window_ends(len(y4), y4, train=True)
    near = tr[(tr >= 100 - FPS) & (tr <= 115 + FPS)]
    far = tr[tr < 100 - FPS]
    assert (np.diff(near) <= 2).all() and (np.diff(far) == 8).all()


def test_clip_windows_slices_and_labels() -> None:
    y4 = _labels((0, 60), (1, 15), (2, 60))
    n = len(y4)
    kp = np.arange(n, dtype=np.float32)[:, None, None] * np.ones((1, 13, 3), np.float32)
    present, ctx = _all_visible(n)
    ends = windows.window_ends(n, y4, train=False)
    out = windows.clip_windows(kp, present, ctx, y4, ends, CAP)
    assert out["kp"].shape == (len(ends), 30, 13, 3)
    # window content is exactly the 30 frames ending at end_frame
    np.testing.assert_array_equal(out["kp"][:, -1, 0, 0], out["end_frame"])
    np.testing.assert_array_equal(out["kp"][:, 0, 0, 0], out["end_frame"] - 29)
    assert set(out["y"]) == {0, 1, 2}


def test_window_crossing_clip_boundary_raises() -> None:
    y4 = _labels((0, 40))
    present, ctx = _all_visible(40)
    with pytest.raises(ValueError, match="clip boundary"):
        windows.clip_windows(
            np.zeros((40, 13, 3)), present, ctx, y4, np.array([10]), CAP
        )


def _index(rows: list[tuple[str, str, int, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["dataset", "path", "subject", "split"])


def test_split_check_passes_clean_index() -> None:
    windows.check_splits(
        _index(
            [
                ("le2i", "a", 1, "train"),
                ("le2i", "b", 2, "test"),
                ("le2i", "c", -1, "train"),
                ("le2i", "d", -1, "test"),
                ("mcfd", "e", 1, "train"),
                ("mcfd", "f", 1, "test"),
            ]
        )
    )


def test_split_check_catches_person_leak() -> None:
    with pytest.raises(ValueError, match="train and test"):
        windows.check_splits(
            _index([("le2i", "a", 1, "train"), ("le2i", "b", 1, "test")])
        )


def test_split_check_catches_clip_in_two_splits() -> None:
    with pytest.raises(ValueError, match="two splits"):
        windows.check_splits(
            _index([("le2i", "a", 1, "train"), ("le2i", "a", 1, "val")])
        )


def test_select_keypoints_to_pixels() -> None:
    kp33 = np.full((3, 33, 4), 0.5, np.float32)
    detected = np.array([True, False, True])
    kp = windows.select_keypoints(kp33, detected, 640, 480)
    assert kp.shape == (3, 13, 3)
    np.testing.assert_allclose(kp[0, 0], [320, 240, 0.5])
    assert np.isnan(kp[1]).all()


def test_fallback_context() -> None:
    n = 40
    kp = np.full((n, 13, 3), np.nan, np.float32)
    kp[:20, :, 0] = 320
    kp[:20, :, 1] = np.linspace(200, 300, 20)[:, None]  # hips moving down
    present = np.arange(n) < 20
    ctx = windows._fallback_last_seen_context(
        kp, present, np.array([640, 480]), FPS, CAP
    )
    assert ctx.shape == (n, 4) and np.isfinite(ctx).all()
    assert (ctx[:20, 0] == 0).all()
    np.testing.assert_allclose(ctx[20:, 0], np.arange(1, 21) / FPS, rtol=1e-5)
    assert ctx[25, 2] > 0  # last speed: moving down
    np.testing.assert_allclose(ctx[25, 1], 300 / 480, rtol=1e-5)
