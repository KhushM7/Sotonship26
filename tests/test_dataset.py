from pathlib import Path

import numpy as np
import pytest
import torch

from fallsense import dataset, windows
from fallsense.features import N_FEATURES, make_features

WH = np.array([640, 480])
N_CLIP = 200


def _standing_clip(n: int) -> np.ndarray:
    ys = np.array(
        [100, 150, 150, 200, 200, 250, 250, 260, 260, 340, 340, 420, 420], float
    )
    xs = np.array(
        [320, 300, 340, 295, 345, 292, 348, 305, 335, 305, 335, 305, 335], float
    )
    kp = np.stack([xs, ys, np.full(13, 0.9)], axis=1)
    return np.repeat(kp[None], n, axis=0).astype(np.float32)


def _fake_simulate(
    kp: np.ndarray,
    present: np.ndarray,
    wh: np.ndarray,
    rng: np.random.Generator,
    onset: int,
    mode: str,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Stand-in for occlusion.simulate_occlusion: hide everything from `onset` on."""
    assert mode in ("rise", "box")
    kp, present = kp.copy(), present.copy()
    kp[onset:] = np.nan
    present[onset:] = False
    return kp, present


def _bank() -> dataset.ClipBank:
    y4 = np.zeros(N_CLIP, np.int8)
    y4[120:135] = 1
    y4[135:] = 2
    return dataset.ClipBank(
        {
            0: {
                "kp": _standing_clip(N_CLIP),
                "present": np.ones(N_CLIP, bool),
                "y4": y4,
                "wh": WH,
            }
        }
    )


def test_augment_keeps_shapes_and_features_finite() -> None:
    rng = np.random.default_rng(0)
    kp = _standing_clip(30)
    present, ctx = np.ones(30, bool), np.zeros((30, 4), np.float32)
    for _ in range(200):
        a_kp, a_present, a_ctx = dataset.augment_kp(kp, present, ctx, WH, rng)
        assert (
            a_kp.shape == (30, 13, 3)
            and a_present.shape == (30,)
            and a_ctx.shape == (30, 4)
        )
        assert np.isfinite(make_features(a_kp, WH, a_present, a_ctx)).all()


def test_augment_does_not_touch_input() -> None:
    kp = _standing_clip(30)
    before = kp.copy()
    dataset.augment_kp(
        kp, np.ones(30, bool), np.zeros((30, 4)), WH, np.random.default_rng(1)
    )
    np.testing.assert_array_equal(kp, before)


def test_speed_change_rate_one_is_identity() -> None:
    kp = np.random.default_rng(2).random((30, 13, 3))
    present, ctx = np.ones(30, bool), np.random.default_rng(3).random((30, 4))
    k2, p2, c2 = dataset._speed_change(kp, present, ctx, 1.0)
    np.testing.assert_allclose(k2, kp)
    np.testing.assert_allclose(c2, ctx)
    assert (p2 == present).all()


def test_occlusion_probability_bounds() -> None:
    rng = np.random.default_rng(4)
    never = {**dataset.OCC, "aug_p_fall": 0.0, "aug_p_other": 0.0}
    always = {**dataset.OCC, "aug_p_fall": 1.0, "aug_p_other": 1.0}
    assert dataset.maybe_occlude(_bank(), 0, 150, rng, _fake_simulate, never) is None
    out = dataset.maybe_occlude(_bank(), 0, 150, rng, _fake_simulate, always)
    assert out is not None
    kp, present, ctx = out
    assert kp.shape == (30, 13, 3) and present.shape == (30,) and ctx.shape == (30, 4)


def test_fall_onset_is_after_fall_start() -> None:
    always = {**dataset.OCC, "aug_p_fall": 1.0, "aug_p_other": 1.0}
    rng = np.random.default_rng(5)
    for _ in range(50):
        _kp, present, ctx = dataset.maybe_occlude(
            _bank(), 0, 199, rng, _fake_simulate, always
        )
        # onset is never before the fall start (frame 120), so at most frames
        # 120-199 are hidden and seconds-since-seen never exceeds 80 frames
        hidden = ~present
        if hidden.any():
            first = np.flatnonzero(hidden)[0]
            assert ctx[first, 0] > 0
            assert ctx[-1, 0] <= (199 - 120 + 1) / dataset.FPS + 1e-4


def _toy_npz(tmp_path: Path, n: int = 8) -> Path:
    kp = np.repeat(_standing_clip(30)[None], n, axis=0)
    path = tmp_path / "train.npz"
    windows.save_windows(
        "train",
        kp=kp,
        wh=np.tile(WH, (n, 1)),
        y=np.arange(n) % 4,
        clip_id=np.zeros(n),
        end_frame=np.linspace(29, 199, n).astype(int),
        present=np.ones((n, 30), bool),
        ctx=np.zeros((n, 30, 4)),
        path=path,
    )
    return path


def test_window_dataset_items(tmp_path: Path) -> None:
    path = _toy_npz(tmp_path)
    stats = {
        "mu": np.zeros(N_FEATURES, np.float32),
        "sd": np.ones(N_FEATURES, np.float32),
    }
    ds = dataset.WindowDataset(
        path, stats, augment=True, bank=_bank(), simulate=_fake_simulate
    )
    assert len(ds) == 8
    x, y = ds[3]
    assert isinstance(x, torch.Tensor) and x.shape == (30, N_FEATURES) and y == 3
    assert torch.isfinite(x).all()


def test_window_features_matches_make_features(tmp_path: Path) -> None:
    d = windows.load_windows(None, path=_toy_npz(tmp_path))
    f = dataset.window_features(d)
    assert f.shape == (8, 30, N_FEATURES)
    np.testing.assert_allclose(
        f[0], make_features(d["kp"][0], d["wh"][0], d["present"][0], d["ctx"][0])
    )


def test_bank_without_occlusion_module_is_clear_error(tmp_path: Path) -> None:
    try:
        import fallsense.occlusion  # noqa: F401
    except ImportError:
        stats = {"mu": np.zeros(N_FEATURES), "sd": np.ones(N_FEATURES)}
        with pytest.raises(RuntimeError, match="occlusion.py"):
            dataset.WindowDataset(_toy_npz(tmp_path), stats, augment=True, bank=_bank())
    else:
        pytest.skip("Dev A's occlusion.py is merged")
