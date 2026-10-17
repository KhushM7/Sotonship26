"""Environment check: prints OK/FAIL per line, exits non-zero if any check fails."""

import importlib
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

MODELS = [
    REPO / "models" / "pose" / "pose_landmarker_full.task",
    REPO / "models" / "pose" / "pose_landmarker_lite.task",
]
ENV_VARS = ["FALL_DATA", "OMNIFALL_CACHE_DIR", "OMNIFALL_DOWNLOAD_DIR", "OMNIFALL_ROOT"]
MIN_FREE_GB = 10

failures = 0


def report(ok, msg):
    global failures
    if not ok:
        failures += 1
    print(f"{'OK  ' if ok else 'FAIL'} {msg}")


def check_python():
    v = sys.version_info
    report((v.major, v.minor) == (3, 11), f"Python 3.11 (found {v.major}.{v.minor}.{v.micro})")


def check_imports():
    for name in ["mediapipe", "cv2", "torch", "omnifall", "fallsense"]:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "")
            extra = f" {ver}" if name in ("mediapipe", "torch") and ver else ""
            report(True, f"import {name}{extra}")
        except Exception as e:
            report(False, f"import {name} ({type(e).__name__}: {e})")


def check_ffmpeg():
    path = shutil.which("ffmpeg")
    report(path is not None, f"ffmpeg on PATH ({path})")


def check_env_vars():
    for var in ENV_VARS:
        report(bool(os.environ.get(var)), f"env {var} set")
    root = os.environ.get("OMNIFALL_ROOT")
    report(bool(root) and Path(root).exists(), f"OMNIFALL_ROOT exists ({root})")


def check_disk():
    free = shutil.disk_usage(REPO.anchor or ".").free / 1e9
    report(free >= MIN_FREE_GB, f"at least {MIN_FREE_GB} GB free ({free:.1f} GB)")


def check_models():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except Exception as e:
        report(False, f"pose models: mediapipe unavailable ({e})")
        return
    for m in MODELS:
        try:
            opts = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(m))
            )
            with vision.PoseLandmarker.create_from_options(opts):
                pass
            report(True, f"PoseLandmarker loads {m.name}")
        except Exception as e:
            report(False, f"PoseLandmarker loads {m.name} ({type(e).__name__}: {e})")


def check_webcam():
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        cap = cv2.VideoCapture(0)
        frame = None
        try:
            # first frames are often black while the camera warms up
            for _ in range(10):
                ok, frame = cap.read()
            if not ok or frame is None:
                report(False, "webcam frame (VideoCapture(0) returned no frame)")
                return
        finally:
            cap.release()
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODELS[0]))
        )
        with vision.PoseLandmarker.create_from_options(opts) as lm:
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = lm.detect(img)
        report(len(res.pose_landmarks) > 0, "webcam frame has a pose (stand in view)")
    except Exception as e:
        report(False, f"webcam pose ({type(e).__name__}: {e})")


def print_threads():
    try:
        import torch

        print(f"INFO torch.get_num_threads() = {torch.get_num_threads()}")
    except Exception as e:
        report(False, f"torch threads ({e})")


def main():
    check_python()
    check_imports()
    check_ffmpeg()
    check_env_vars()
    check_disk()
    check_models()
    check_webcam()
    print_threads()
    print(f"\n{'All checks passed' if not failures else f'{failures} check(s) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
