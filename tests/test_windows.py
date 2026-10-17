import numpy as np
import pytest

from fallsense import windows


def _toy(n=5):
    return dict(
        kp=np.random.rand(n, 30, 13, 3),
        wh=np.tile([640, 480], (n, 1)),
        y=np.arange(n) % 4,
        clip_id=np.arange(n),
        end_frame=np.arange(n) * 3 + 29,
    )


@pytest.fixture
def tmp_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(windows, "windows_path", lambda split: tmp_path / f"{split}.npz")


def test_round_trip_dtypes_and_shapes(tmp_windows):
    windows.save_windows("val", **_toy())
    d = windows.load_windows("val")
    for key, (dtype, tail) in windows.SCHEMA.items():
        assert d[key].dtype == dtype
        assert d[key].shape == (5, *tail)


def test_empty_split_is_valid(tmp_windows):
    windows.save_windows("oops", **_toy(0))
    assert len(windows.load_windows("oops")["y"]) == 0


def test_bad_label_rejected(tmp_windows):
    t = _toy()
    t["y"][0] = 4
    with pytest.raises(ValueError, match="y must be"):
        windows.save_windows("train", **t)


def test_bad_shape_rejected(tmp_windows):
    t = _toy()
    t["kp"] = t["kp"][:, :20]
    with pytest.raises(ValueError, match="kp: shape"):
        windows.save_windows("train", **t)
