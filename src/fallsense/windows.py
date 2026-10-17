"""Contract 3 v2: window arrays, data/windows/{split}.npz."""

from typing import Any, cast

import numpy as np
import numpy.typing as npt

from fallsense.config import windows_path

N_CLASSES = 4
WINDOW = 30
N_KEYPOINTS = 13
N_CTX = 4  # occlusion.last_seen_context columns

# key -> (dtype, shape after the leading N axis)
SCHEMA = {
    "kp": (np.float32, (WINDOW, N_KEYPOINTS, 3)),
    "wh": (np.int16, (2,)),
    "y": (np.int64, ()),
    "clip_id": (np.int32, ()),
    "end_frame": (np.int32, ()),
    "present": (np.bool_, (WINDOW,)),
    "ctx": (np.float32, (WINDOW, N_CTX)),
}


def validate_windows(arrays: dict[str, npt.NDArray[Any]]) -> None:
    """Raise ValueError unless `arrays` matches Contract 3."""
    missing = set(SCHEMA) - set(arrays)
    if missing:
        raise ValueError(f"missing keys: {sorted(missing)}")
    n = len(arrays["y"])
    for key, (dtype, tail) in SCHEMA.items():
        a = arrays[key]
        if a.dtype != dtype:
            raise ValueError(f"{key}: dtype {a.dtype}, expected {np.dtype(dtype)}")
        if a.shape != (n, *tail):
            raise ValueError(f"{key}: shape {a.shape}, expected {(n, *tail)}")
    if n and not ((arrays["y"] >= 0) & (arrays["y"] < N_CLASSES)).all():
        raise ValueError(f"y must be in 0..{N_CLASSES - 1}")
    if not np.isfinite(arrays["ctx"]).all():
        raise ValueError("ctx must be finite")


def save_windows(split, kp, wh, y, clip_id, end_frame, present, ctx) -> None:
    """Cast to the contract dtypes, validate and write data/windows/{split}.npz."""
    arrays = {
        key: np.asarray(val, dtype=SCHEMA[key][0])
        for key, val in zip(SCHEMA, (kp, wh, y, clip_id, end_frame, present, ctx))
    }
    validate_windows(arrays)
    path = windows_path(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **cast("dict[str, Any]", arrays))


def load_windows(split: str) -> dict[str, npt.NDArray[Any]]:
    """Load and validate data/windows/{split}.npz."""
    with np.load(windows_path(split)) as d:
        arrays = {key: d[key] for key in SCHEMA}
    validate_windows(arrays)
    return arrays
