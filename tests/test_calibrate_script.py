import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "07_calibrate.py"
_spec = importlib.util.spec_from_file_location("calibrate_script", _PATH)
assert _spec is not None and _spec.loader is not None
cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cal)


def _sample(true_t: float, n: int = 20000, seed: int = 0):
    """Labels drawn from softmax(z), logits reported as z * true_t."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 2, size=(n, 4))
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(4, p=row) for row in p])
    return z * true_t, y


@pytest.mark.parametrize("true_t", [0.5, 2.0])
def test_temperature_recovers_overconfidence(true_t: float) -> None:
    logits, y = _sample(true_t)
    assert cal.fit_temperature(logits, y) == pytest.approx(true_t, rel=0.08)


def test_temperature_lowers_ece() -> None:
    logits, y = _sample(3.0)
    t = cal.fit_temperature(logits, y)
    soft = lambda T: np.exp(logits / T) / np.exp(logits / T).sum(1, keepdims=True)
    assert cal.ece(soft(t), y) < cal.ece(soft(1.0), y)
    assert cal.ece(soft(t), y) < 0.02


def test_pick_t_high_takes_highest_meeting_target() -> None:
    table = pd.DataFrame(
        {"t": [0.1, 0.2, 0.3, 0.4], "sens_fall": [0.98, 0.93, 0.88, 0.70]}
    )
    assert cal.pick_t_high(table, 0.90) == (0.2, True)
    # nothing reaches the target: highest threshold near the best sensitivity
    assert cal.pick_t_high(table, 0.99, plateau=0.0) == (0.1, False)
    assert cal.pick_t_high(table, 0.99, plateau=0.05) == (0.2, False)


def test_pick_t_high_unreachable_prefers_fewest_alerts() -> None:
    table = pd.DataFrame(
        {"t": [0.05, 0.1, 0.15, 0.2], "sens_fall": [0.77, 0.77, 0.70, 0.60]}
    )
    assert cal.pick_t_high(table, 0.90, plateau=0.0) == (0.1, False)
    # 0.70 is within 0.08 of the best (0.77), so the higher threshold wins
    assert cal.pick_t_high(table, 0.90, plateau=0.08) == (0.15, False)
