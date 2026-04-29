"""MediaPipe Pose Landmarker wrapper.

Produces a per-frame (33, 4) landmark array from an MP4. The 4 channels are
(x, y, z, visibility). Coordinates are in MediaPipe's normalized image space:
x and y in [0, 1] relative to the frame, z in image-space units (smaller =
closer to camera). Visibility in [0, 1].

Usage:
    extractor = PoseExtractor()
    landmarks, meta = extractor.extract(Path("clip.mp4"))
    # landmarks: np.ndarray of shape (T, 33, 4)
    # meta: {"fps": 30.0, "num_frames": 91, "width": 1080, "height": 1920}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PoseMeta:
    fps: float
    num_frames: int
    width: int
    height: int
    detected_frames: int = 0
    """Number of frames where MediaPipe successfully detected a pose."""

    def asdict(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "num_frames": self.num_frames,
            "width": self.width,
            "height": self.height,
            "detected_frames": self.detected_frames,
        }


@dataclass
class PoseExtractor:
    """Wraps mediapipe.solutions.pose.

    `model_complexity` ∈ {0, 1, 2} — 0 is fastest, 2 most accurate. We default
    to 1 which matches MediaPipe's recommended balance and runs cleanly on
    Modal's CPU containers.
    """
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    frame_skip: int = 1
    """Process every Nth frame (1 = every frame)."""

    _pose: Any = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._pose is not None:
            return
        import mediapipe as mp  # local import: heavy dependency

        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            enable_segmentation=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
            self._pose = None

    def extract(self, video_path: Path) -> tuple[np.ndarray, PoseMeta]:
        import cv2  # local import: heavy dependency

        if not video_path.exists():
            raise FileNotFoundError(video_path)

        self._load()
        assert self._pose is not None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cv2 could not open: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames: list[np.ndarray] = []
        detected = 0
        idx = 0
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if idx % self.frame_skip != 0:
                    idx += 1
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                result = self._pose.process(frame_rgb)
                if result.pose_landmarks is not None:
                    detected += 1
                    arr = np.asarray(
                        [
                            (lm.x, lm.y, lm.z, lm.visibility)
                            for lm in result.pose_landmarks.landmark
                        ],
                        dtype=np.float32,
                    )
                else:
                    # Carry the previous frame if available, else zeros.
                    arr = frames[-1] if frames else np.zeros((33, 4), dtype=np.float32)
                frames.append(arr)
                idx += 1
        finally:
            cap.release()

        if not frames:
            raise RuntimeError(f"no frames extracted from: {video_path}")

        landmarks = np.stack(frames, axis=0)
        meta = PoseMeta(
            fps=fps / max(self.frame_skip, 1),
            num_frames=len(frames),
            width=width,
            height=height,
            detected_frames=detected,
        )
        return landmarks, meta
