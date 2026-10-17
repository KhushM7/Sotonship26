"""MediaPipe Pose Landmarker wrapper producing Contract 1 arrays per frame."""

import os
from typing import Self

import cv2
import mediapipe as mp
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike

V = mp.tasks.vision

F32 = npt.NDArray[np.float32]


class PoseExtractor:
    """One instance per clip (VIDEO mode needs strictly increasing timestamps)."""

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        min_det: float = 0.5,
        min_presence: float = 0.5,
        min_track: float = 0.5,
        upscale: float = 1.0,
    ) -> None:
        opts = V.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=V.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_det,
            min_pose_presence_confidence=min_presence,
            min_tracking_confidence=min_track,
        )
        self.lm = V.PoseLandmarker.create_from_options(opts)
        self.upscale = upscale

    def __call__(self, frame_bgr: MatLike, t_ms: int) -> tuple[F32, F32, bool]:
        """Return (kp (33, 4), world (33, 3), detected) for one frame."""
        if self.upscale != 1.0:
            # Landmarks are image-normalised, so upscaling does not change their scale.
            frame_bgr = cv2.resize(
                frame_bgr,
                None,
                fx=self.upscale,
                fy=self.upscale,
                interpolation=cv2.INTER_LINEAR,
            )
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # MediaPipe wants RGB
        res = self.lm.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t_ms
        )
        if not res.pose_landmarks:
            return (
                np.full((33, 4), np.nan, np.float32),
                np.full((33, 3), np.nan, np.float32),
                False,
            )
        kp = np.array(
            [[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks[0]], np.float32
        )
        w = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]], np.float32)
        return kp, w, True

    def close(self) -> None:
        self.lm.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
