from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from fallsense import calibrate, scores, windows
from fallsense.features import N_FEATURES
from fallsense.models import build_model

TH = {"t_high": 0.8, "t_low": 0.5}
DT = 3 / 15  # eval stride: 0.2 s between windows


def _scores(
    p_falling: list[float] | np.ndarray,
    p_fallen: list[float] | np.ndarray | None = None,
    clip_id: int = 0,
) -> pd.DataFrame:
    """Contract 4 rows for one clip, one window every 0.2 s."""
    pf = np.asarray(p_falling, float)
    pd_ = np.zeros_like(pf) if p_fallen is None else np.asarray(p_fallen, float)
    n = len(pf)
    end = 29 + 3 * np.arange(n)
    rest = 1 - pf - pd_
    return pd.DataFrame(
        {
            "clip_id": np.full(n, clip_id),
            "end_frame": end,
            "t_s": end / 15,
            "p_upright": rest,
            "p_falling": pf,
            "p_fallen": pd_,
            "p_controlled_low": np.zeros(n),
        }
    )


def _levels(alerts: pd.DataFrame) -> list[str]:
    return alerts["level"].tolist()


def test_quiet_clip_gives_no_alerts() -> None:
    a = calibrate.alerts_from_scores(_scores([0.1] * 50), TH)
    assert len(a) == 0
    assert list(a.columns) == calibrate.ALERT_COLUMNS


def test_spike_gives_one_fall_at_the_spike() -> None:
    pf = [0.1] * 50
    pf[20] = 0.9
    a = calibrate.alerts_from_scores(_scores(pf), TH)
    assert _levels(a) == ["FALL"]
    assert a["t_s"].iloc[0] == pytest.approx((29 + 3 * 20) / 15)
    assert a["score"].iloc[0] == pytest.approx(0.9)


def test_middle_band_is_check() -> None:
    pf = [0.1] * 50
    pf[10] = 0.6
    assert _levels(calibrate.alerts_from_scores(_scores(pf), TH)) == ["CHECK"]


def test_fall_score_is_max_over_last_three_windows() -> None:
    pf = [0.1] * 50
    pf[10] = 0.6
    pf[12] = 0.4  # 0.6 is still inside the last 3 windows here: no new alert
    s = _scores(pf)
    a = calibrate.alerts_from_scores(s, TH)
    assert len(a) == 1
    # the 3-window max holds a lone spike for exactly 3 windows
    rolled = s["p_falling"].rolling(3, min_periods=1).max()
    assert (rolled >= 0.6).sum() == 3


def test_check_escalates_to_fall_inside_refractory() -> None:
    pf = [0.1] * 50
    pf[10], pf[12] = 0.6, 0.9
    assert _levels(calibrate.alerts_from_scores(_scores(pf), TH)) == ["CHECK", "FALL"]


def test_refractory_suppresses_repeats_within_30_s() -> None:
    n = int(70 / DT)
    pf = np.full(n, 0.1)
    for t in (5, 15, 34, 40):  # 10 s and 29 s after the first: dropped
        pf[int(t / DT)] = 0.9  # 34 s: fires, 40 s: inside its refractory
    a = calibrate.alerts_from_scores(_scores(pf), TH)
    assert _levels(a) == ["FALL", "FALL"]
    assert np.diff(a["t_s"].to_numpy())[0] >= 30


def test_fall_after_fall_then_check_is_quiet() -> None:
    pf = [0.1] * 60
    pf[10], pf[30] = 0.9, 0.6  # a later CHECK is lower than the FALL: suppressed
    assert _levels(calibrate.alerts_from_scores(_scores(pf), TH)) == ["FALL"]


def test_on_floor_after_continuous_fallen() -> None:
    n = 60  # 12 s
    pf = np.full(n, 0.05)
    pfl = np.zeros(n)
    pfl[10:] = 0.7  # fallen from window 10 on
    a = calibrate.alerts_from_scores(_scores(pf, pfl), TH)
    assert _levels(a) == ["ON FLOOR"]
    t0 = (29 + 3 * 10) / 15
    assert a["t_s"].iloc[0] == pytest.approx(t0 + 5, abs=DT)
    # demo setting: 20 s on the floor is not reached in 12 s
    a20 = calibrate.alerts_from_scores(_scores(pf, pfl), TH, on_floor_seconds=20)
    assert len(a20) == 0


def test_on_floor_reads_thresholds_and_needs_continuity() -> None:
    n = 100
    pf = np.full(n, 0.05)
    pfl = np.full(n, 0.7)
    pfl[::20] = 0.2  # breaks every 4 s: never 5 s continuous
    assert len(calibrate.alerts_from_scores(_scores(pf, pfl), TH)) == 0
    th = {**TH, "on_floor_seconds_eval": 2.0}
    # every 4 s run passes 2 s, but the 30 s refractory keeps only the first
    assert _levels(calibrate.alerts_from_scores(_scores(pf, pfl), th)) == ["ON FLOOR"]
    # once per continuous run: a 20 s run fires once
    pfl2 = np.full(n, 0.7)
    assert len(calibrate.alerts_from_scores(_scores(pf, pfl2), TH)) == 1


def test_fall_escalates_to_on_floor() -> None:
    n = 60
    pf = np.full(n, 0.05)
    pf[5] = 0.95
    pfl = np.zeros(n)
    pfl[6:] = 0.8
    a = calibrate.alerts_from_scores(_scores(pf, pfl), TH)
    assert _levels(a) == ["FALL", "ON FLOOR"]


def test_clips_are_independent() -> None:
    pf = [0.1] * 20
    pf[5] = 0.9
    s = pd.concat([_scores(pf, clip_id=3), _scores(pf, clip_id=1)])
    a = calibrate.alerts_from_scores(s.sample(frac=1, random_state=0), TH)
    assert a["clip_id"].tolist() == [1, 3]
    assert _levels(a) == ["FALL", "FALL"]


# ---------------------------------------------------------------- write_scores


def _toy_run(tmp_path: Path) -> tuple[Path, Path]:
    torch.manual_seed(0)
    run = tmp_path / "toy-run"
    run.mkdir()
    torch.save(
        {"model": "gru", "state_dict": build_model("gru").state_dict()},
        run / "model.pt",
    )
    np.savez(
        run / "feat_stats.npz",
        mu=np.zeros(N_FEATURES, np.float32),
        sd=np.ones(N_FEATURES, np.float32),
    )
    n = 12
    rng = np.random.default_rng(0)
    kp = rng.uniform(100, 400, size=(n, 30, 13, 3)).astype(np.float32)
    kp[..., 2] = 0.9
    split = tmp_path / "val.npz"
    windows.save_windows(
        "val",
        kp=kp,
        wh=np.tile([640, 480], (n, 1)),
        y=np.arange(n) % 4,
        clip_id=np.repeat([2, 0], n // 2),
        end_frame=np.tile(29 + 3 * np.arange(n // 2), 2),
        present=np.ones((n, 30), bool),
        ctx=np.zeros((n, 30, 4)),
        path=split,
    )
    return run, split


def test_write_scores_writes_contract_4(tmp_path: Path) -> None:
    run, split = _toy_run(tmp_path)
    out = tmp_path / "scores" / "val.parquet"
    df = calibrate.write_scores(run, split, out)
    back = pd.read_parquet(out)
    scores.validate_scores(back)
    assert len(back) == 12
    assert back["clip_id"].tolist() == sorted(back["clip_id"].tolist())
    np.testing.assert_allclose(back["t_s"], back["end_frame"] / 15, rtol=1e-6)
    np.testing.assert_allclose(back["p_falling"], df["p_falling"], rtol=1e-6)


def test_write_scores_temperature(tmp_path: Path) -> None:
    run, split = _toy_run(tmp_path)
    p1 = calibrate.write_scores(run, split, tmp_path / "t1.parquet")
    (run / "thresholds.yaml").write_text("temperature: 100.0\n", encoding="utf-8")
    p100 = calibrate.write_scores(run, split, tmp_path / "t100.parquet")
    cols = calibrate.CLASS_COLUMNS
    # a high temperature pushes every row towards uniform
    spread1 = p1[cols].to_numpy().std(axis=1)
    spread100 = p100[cols].to_numpy().std(axis=1)
    assert (spread100 < spread1).all()
    assert np.allclose(p100[cols].to_numpy(), 0.25, atol=0.01)


# ---------------------------------------------------------------- ensembles


def test_ensemble_logits_softmax_is_mean_of_members() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(5, 4)), rng.normal(size=(5, 4))
    got = calibrate.softmax_t(calibrate.ensemble_logits([a, b]))
    want = (calibrate.softmax_t(a) + calibrate.softmax_t(b)) / 2
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_ensemble_of_one_run_scores_like_the_run(tmp_path: Path) -> None:
    run, split = _toy_run(tmp_path)
    ens = tmp_path / "toy-ens"
    ens.mkdir()
    (ens / "ensemble.yaml").write_text("members: [toy-run, toy-run]\n", "utf-8")
    single = calibrate.write_scores(run, split, tmp_path / "s.parquet")
    double = calibrate.write_scores(ens, split, tmp_path / "e.parquet")
    cols = calibrate.CLASS_COLUMNS
    np.testing.assert_allclose(double[cols], single[cols], atol=1e-5)


def test_alert_state_matches_alerts_from_scores() -> None:
    pf = [0.1, 0.9, 0.2, 0.6, 0.1] + [0.0] * 200 + [0.95]
    fallen = [0.0] * 3 + [0.9] * 40 + [0.0] * 163
    df = _scores(pf, fallen)
    state = calibrate.AlertState(TH["t_high"], TH["t_low"], on_floor_s=5)
    live = [
        (t, *hit)
        for t, p, f in zip(df["t_s"], df["p_falling"], df["p_fallen"])
        if (hit := state.step(t, p, f)) is not None
    ]
    a = calibrate.alerts_from_scores(df, {**TH, "on_floor_seconds_eval": 5})
    assert [lv for _, lv, _ in live] == _levels(a)
    np.testing.assert_allclose([t for t, _, _ in live], a["t_s"], rtol=1e-6)
