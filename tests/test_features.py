import numpy as np

from fallsense import features as F

W, H = 640, 480
WH = np.array([W, H])
# a standing person, pixel (x, y) per config keypoint; hip midpoint at (320, 260)
STANDING = np.array(
    [
        [320, 100],  # nose
        [300, 150],  # shoulders
        [340, 150],
        [295, 200],  # elbows
        [345, 200],
        [292, 250],  # wrists
        [348, 250],
        [305, 260],  # hips
        [335, 260],
        [305, 340],  # knees
        [335, 340],
        [305, 420],  # ankles
        [335, 420],
    ],
    dtype=np.float64,
)
HIP = np.array([320.0, 260.0])


def _pose(theta: float) -> np.ndarray:
    """Standing template rotated by `theta` (0 upright, pi/2 lying) about the hips."""
    c, s = np.cos(theta), np.sin(theta)
    xy = (STANDING - HIP) @ np.array([[c, s], [-s, c]]) + HIP
    return np.concatenate([xy, np.full((13, 1), 0.9)], axis=1)


def _clip(thetas: np.ndarray) -> np.ndarray:
    return np.stack([_pose(t) for t in thetas])


def _ctx_stub(
    kp: np.ndarray, present: np.ndarray, cap_s: float = F.CTX_CAP_S, fps: float = F.FPS
) -> np.ndarray:
    """Stand-in for occlusion.last_seen_context until M19 lands."""
    T = len(present)
    ctx = np.zeros((T, 4))
    since, last = cap_s, np.array([cap_s, 0.0, 0.0, 0.0])
    for t in range(T):
        if present[t]:
            hip_y = kp[t, [F.L_HIP, F.R_HIP], 1].mean() / H
            since = 0.0
            last = np.array([0.0, hip_y, 0.0, min(hip_y, 1 - hip_y)])
        else:
            since = min(since + 1 / fps, cap_s)
        ctx[t] = [since, *last[1:]]
    return ctx


def _features(kp: np.ndarray, present: np.ndarray | None = None) -> np.ndarray:
    present = np.isfinite(kp[..., 0]).any(axis=1) if present is None else present
    return F.make_features(kp, WH, present, _ctx_stub(kp, present))


def _col(name: str) -> int:
    return F.FEATURE_NAMES.index(name)


def test_names_and_shape() -> None:
    assert len(F.FEATURE_NAMES) == 51 == len(set(F.FEATURE_NAMES))
    x = _features(_clip(np.zeros(30)))
    assert x.shape == (30, 51) and x.dtype == np.float32
    assert {"log_bbox_aspect", "hip_height"} <= set(F.FEATURE_NAMES)


def test_vanished_tail_is_finite_and_carries_ctx() -> None:
    kp = _clip(np.zeros(30))
    kp[-10:] = np.nan
    present = np.arange(30) < 20
    ctx = _ctx_stub(kp, present)
    x = F.make_features(kp, WH, present, ctx)
    assert np.isfinite(x).all()
    np.testing.assert_allclose(
        x[:, _col("ctx_since_seen")], ctx[:, 0] / F.CTX_CAP_S, atol=1e-6
    )
    for i, name in enumerate(F.CTX_NAMES[1:], start=1):
        np.testing.assert_allclose(x[:, _col(name)], ctx[:, i], atol=1e-6)
    assert x[-1, _col("detected")] == 0 and x[0, _col("detected")] == 1
    assert x[-1, _col("ctx_since_seen")] > 0


def test_doubling_coordinates_keeps_shape() -> None:
    kp = _clip(np.linspace(0, 1.2, 30))
    big = kp.copy()
    big[..., :2] *= 2
    np.testing.assert_allclose(_features(kp)[:, :26], _features(big)[:, :26], atol=1e-5)


def test_mirror_keeps_torso_angle_and_hip_height() -> None:
    kp = _clip(np.linspace(0, 1.2, 30))
    mirrored = kp.copy()
    mirrored[..., 0] = W - mirrored[..., 0]
    a, b = _features(kp), _features(mirrored)
    for name in ("torso_angle", "hip_height"):
        np.testing.assert_allclose(a[:, _col(name)], b[:, _col(name)], atol=1e-5)


def test_forty_percent_nan_frames_finite() -> None:
    rng = np.random.default_rng(0)
    kp = _clip(np.linspace(0, 1.5, 30))
    kp[rng.choice(30, 12, replace=False)] = np.nan
    assert np.isfinite(_features(kp)).all()


def test_standing_then_lying_torso_angle() -> None:
    thetas = np.concatenate(
        [np.zeros(10), np.linspace(0, np.pi / 2, 10), np.full(10, np.pi / 2)]
    )
    angle = _features(_clip(thetas))[:, _col("torso_angle")]
    assert angle[:10].max() < 0.05
    assert abs(angle[-10:].mean() - 0.5) < 0.05
    hip = _features(_clip(thetas))[:, _col("hip_height")]
    assert hip[0] > 1.0 and abs(hip[-1]) < 0.2


def test_short_gaps_filled_long_gaps_zeroed() -> None:
    kp = _clip(np.zeros(30))
    kp[5:8, F.NOSE] = np.nan  # 3-frame gap: filled
    kp[15:27, F.NOSE] = np.nan  # 12-frame gap: first 5 filled, rest lost
    vis = _features(kp)[:, _col("nose_vis")]
    assert (vis[5:8] > 0).all()
    assert (vis[15:20] > 0).all() and (vis[20:27] == 0).all()
