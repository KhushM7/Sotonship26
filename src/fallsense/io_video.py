"""Read a video as BGR frames resampled to a target frame rate."""

from pathlib import Path

import cv2


def _iter_opencv(path):
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                return
            yield src_fps, frame_bgr
    finally:
        cap.release()


def _iter_pyav(path):
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        src_fps = float(stream.average_rate or 25.0)
        for frame in container.decode(video=0):
            yield src_fps, frame.to_ndarray(format="bgr24")


def opencv_can_read(path) -> bool:
    cap = cv2.VideoCapture(str(path))
    ok, _ = cap.read()
    cap.release()
    return ok


def iter_frames(path, target_fps=15):
    """Yield (t_ms, frame_bgr) for frames kept at roughly target_fps.

    Time is frame index / source fps, not CAP_PROP_POS_MSEC, which is
    unreliable for some containers. Falls back to PyAV when OpenCV cannot
    decode the first frame (e.g. AV1 without a matching codec build).
    """
    path = Path(path)
    frames = _iter_opencv(path) if opencv_can_read(path) else _iter_pyav(path)
    step, next_t = 1.0 / target_fps, 0.0
    for i, (src_fps, frame_bgr) in enumerate(frames):
        if not src_fps or not 1 <= src_fps <= 240:
            src_fps = 25.0
        t = i / src_fps
        if t + 1e-6 >= next_t:
            yield round(t * 1000), frame_bgr
            next_t += step
