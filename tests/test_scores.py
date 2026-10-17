from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fallsense import scores


def _toy(n: int = 6) -> pd.DataFrame:
    p = np.random.dirichlet(np.ones(4), size=n)
    return pd.DataFrame(
        {
            "clip_id": np.arange(n),
            "end_frame": np.arange(n) * 3 + 29,
            "t_s": (np.arange(n) * 3 + 29) / 15,
            "p_upright": p[:, 0],
            "p_falling": p[:, 1],
            "p_fallen": p[:, 2],
            "p_controlled_low": p[:, 3],
        }
    )


@pytest.fixture(autouse=True)
def tmp_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scores, "scores_path", lambda m, s: tmp_path / m / f"{s}.parquet"
    )


def test_round_trip_dtypes(tmp_path: Path) -> None:
    scores.save_scores("e1", "val", _toy())
    df = scores.load_scores("e1", "val")
    assert list(df.columns) == list(scores.COLUMNS)
    assert len(df) == 6
    for col, dtype in scores.COLUMNS.items():
        assert df[col].dtype == dtype


def test_empty_is_valid() -> None:
    scores.save_scores("e1", "oops", _toy(0))
    assert len(scores.load_scores("e1", "oops")) == 0


def test_probabilities_must_sum_to_one() -> None:
    df = _toy()
    df["p_upright"] += 0.5
    with pytest.raises(ValueError):
        scores.save_scores("e1", "val", df)


def test_nan_rejected() -> None:
    df = _toy()
    df.loc[0, "t_s"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        scores.save_scores("e1", "val", df)


def test_missing_column_rejected() -> None:
    with pytest.raises(KeyError):
        scores.save_scores("e1", "val", _toy().drop(columns="p_fallen"))
