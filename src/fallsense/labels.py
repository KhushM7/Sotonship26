"""OmniFall label loading and the 16-class -> 4-class mapping."""

from functools import cache

import pandas as pd

from fallsense.config import load_config

_CFG = load_config()
TEST_ONLY = set(_CFG["test_only_datasets"])


def config_for(dataset: str) -> str:
    """OmniFall config holding a dataset's labels (cross-subject split)."""
    if dataset == "of-syn":
        return "of-syn"
    if dataset == "OOPS":
        return "of-itw"
    return f"{dataset.lower()}{_CFG['split_config_suffix']}"


@cache
def load_labels(dataset: str) -> pd.DataFrame:
    """Every OmniFall segment for one dataset, all splits, with a `split` column.

    `validation` is renamed `val`; test-only datasets (OOPS) put every row in `test`.
    """
    import omnifall

    dd = omnifall.load(config_for(dataset))
    parts = []
    for split, ds in dd.items():
        df = ds.to_pandas()
        df["split"] = "val" if split == "validation" else split
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df = df[df["dataset"] == dataset].reset_index(drop=True)
    if dataset in TEST_ONLY:
        df["split"] = "test"
    return df
