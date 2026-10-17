"""Contract 4: frame scores, reports/scores/{model}/{split}.parquet."""

import numpy as np
import pandas as pd

from fallsense.config import scores_path

COLUMNS = {
    "clip_id": np.int32,
    "end_frame": np.int32,
    "t_s": np.float32,
    "p_upright": np.float32,
    "p_falling": np.float32,
    "p_fallen": np.float32,
    "p_controlled_low": np.float32,
}
PROB_COLUMNS = [c for c in COLUMNS if c.startswith("p_")]


def validate_scores(df: pd.DataFrame) -> None:
    """Raise ValueError unless `df` matches Contract 4."""
    if list(df.columns) != list(COLUMNS):
        raise ValueError(f"columns {list(df.columns)}, expected {list(COLUMNS)}")
    for col, dtype in COLUMNS.items():
        if df[col].dtype != dtype:
            raise ValueError(f"{col}: dtype {df[col].dtype}, expected {np.dtype(dtype)}")
    if df.isna().any().any():
        raise ValueError("NaN in scores")
    probs = df[PROB_COLUMNS].to_numpy()
    if ((probs < 0) | (probs > 1)).any():
        raise ValueError("probabilities must be in [0, 1]")
    if len(df) and not np.allclose(probs.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError("probabilities must sum to 1 per row")


def save_scores(model, split, df: pd.DataFrame) -> None:
    """Cast to the contract dtypes, validate and write the parquet file."""
    df = df[list(COLUMNS)].astype(COLUMNS).reset_index(drop=True)
    validate_scores(df)
    path = scores_path(model, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_scores(model, split) -> pd.DataFrame:
    """Load and validate a score file."""
    df = pd.read_parquet(scores_path(model, split))
    validate_scores(df)
    return df
