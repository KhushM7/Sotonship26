from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from fallsense import windows


def _toy(n=5):
    return dict(
        kp=np.random.rand(n, 30, 13, 3),
        wh=np.tile([640, 480], (n, 1)),
        y=np.arange(n) % 4,
        clip_id=np.arange(n),
        end_frame=np.arange(n) * 3 + 29,
        present=np.random.rand(n, 30) > 0.2,
        ctx=np.random.rand(n, 30, 4),
    )


@pytest.fixture
def tmp_windows(tmp_path, monkeypatch):
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


def test_v2_keys_round_trip(tmp_windows):
    t = _toy()
    windows.save_windows("val", **t)
    d = windows.load_windows("val")
    assert d["present"].dtype == np.bool_ and d["present"].shape == (5, 30)
    assert d["ctx"].dtype == np.float32 and d["ctx"].shape == (5, 30, 4)
    np.testing.assert_array_equal(d["present"], t["present"])
    np.testing.assert_allclose(d["ctx"], t["ctx"], rtol=1e-6)


def test_missing_v2_key_rejected():
    t = {k: np.asarray(v, dtype=windows.SCHEMA[k][0]) for k, v in _toy().items()}
    del t["ctx"]
    with pytest.raises(ValueError, match="missing keys"):
        windows.validate_windows(t)


def test_bad_ctx_shape_rejected(tmp_windows):
    t = _toy()
    t["ctx"] = t["ctx"][..., :3]
    with pytest.raises(ValueError, match="ctx: shape"):
        windows.save_windows("train", **t)


def test_bad_present_shape_rejected(tmp_windows):
    t = _toy()
    t["present"] = t["present"][:, :20]
    with pytest.raises(ValueError, match="present: shape"):
        windows.save_windows("train", **t)


def test_nan_ctx_rejected(tmp_windows):
    t = _toy()
    t["ctx"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="ctx must be finite"):
        windows.save_windows("train", **t)


def test_config_is_schema_v2():
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
