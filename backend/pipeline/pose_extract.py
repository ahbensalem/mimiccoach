"""MediaPipe Pose Landmarker wrapper (Tasks API).

Produces a per-frame (33, 4) landmark array from an MP4. The 4 channels are
(x, y, z, visibility). Coordinates are in MediaPipe's normalized image space:
x and y in [0, 1] relative to the frame, z in image-space units (smaller =
closer to camera). Visibility in [0, 1].

Uses the modern MediaPipe Tasks API (`mediapipe.tasks.python.vision
.PoseLandmarker`). The legacy `mediapipe.solutions.pose` API has been
removed in current mediapipe builds. The Tasks API needs an explicit
`.task` model bundle; we resolve its path from the `MEDIAPIPE_MODEL_PATH`
environment variable (set to `/opt/pose_landmarker.task` inside the
Modal image).

Usage:
    extractor = PoseExtractor()
    landmarks, meta = extractor.extract(Path("clip.mp4"))
    # landmarks: np.ndarray of shape (T, 33, 4)
    # meta: PoseMeta with fps, num_frames, width, height, detected_frames
"""
from __future__ import annotations

import contextlib
import os
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


def _resolve_model_path(explicit: str | os.PathLike[str] | None) -> Path:
    """Pick the .task model bundle.

    Precedence: explicit kwarg > MEDIAPIPE_MODEL_PATH env > /opt/pose_landmarker.task
    """
    if explicit:
        return Path(explicit)
    env = os.environ.get("MEDIAPIPE_MODEL_PATH")
    if env:
        return Path(env)
    return Path("/opt/pose_landmarker.task")


@dataclass
class PoseExtractor:
    """Wraps `mediapipe.tasks.python.vision.PoseLandmarker` in VIDEO mode.

    `min_detection_confidence` and `min_tracking_confidence` map to the
    Tasks API's `min_pose_detection_confidence` and `min_tracking_confidence`.
    `frame_skip > 1` processes every Nth frame to trade accuracy for speed.
    """
    model_path: str | os.PathLike[str] | None = None
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    frame_skip: int = 1

    _landmarker: Any = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._landmarker is not None:
            return
        # Local imports — heavy native deps that we don't want to pull
        # during simple module imports (e.g. unit tests for siblings).
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        path = _resolve_model_path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found at {path}. "
                "Set MEDIAPIPE_MODEL_PATH or pass model_path=… "
                "(downloaded from https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task)."
            )

        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        # Stash the mp module so .extract() can build mp.Image without re-importing.
        self._mp = mp

    def close(self) -> None:
        if self._landmarker is not None:
            with contextlib.suppress(Exception):
                self._landmarker.close()
            self._landmarker = None

    def extract(self, video_path: Path) -> tuple[np.ndarray, PoseMeta]:
        import cv2

        if not video_path.exists():
            raise FileNotFoundError(video_path)

        self._load()
        assert self._landmarker is not None
        mp = self._mp  # cached from _load()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cv2 could not open: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ms_per_frame = 1000.0 / max(fps, 1.0)

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
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                # VIDEO mode requires monotonically increasing timestamps in ms.
                ts_ms = int(idx * ms_per_frame)
                result = self._landmarker.detect_for_video(mp_image, ts_ms)
                if result.pose_landmarks:
                    pose = result.pose_landmarks[0]  # first detected person
                    arr = np.asarray(
                        [(lm.x, lm.y, lm.z, lm.visibility) for lm in pose],
                        dtype=np.float32,
                    )
                    detected += 1
                else:
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
