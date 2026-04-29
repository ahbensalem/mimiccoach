"""Auto-derive a body-type bucket from MediaPipe landmarks.

We classify clips into three buckets — `narrow` / `balanced` / `broad` —
using the ratio of shoulder distance to hip distance, computed on the
clip's most-stable frame (the frame with the lowest joint-velocity
magnitude, treated as a "neutral" pose). Thresholds are loaded from
motions.yaml.

This bucketing is meant to be deterministic and clip-agnostic so we
can apply it uniformly to user uploads and reference clips without
any manual labeling.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from .segment import per_frame_velocity_magnitude
from .skeleton_map import MP_INDEX


@lru_cache(maxsize=1)
def _body_type_config() -> dict[str, Any]:
    from .segment import _config

    return _config()["body_type"]


def shoulder_hip_ratio(landmarks: np.ndarray) -> float:
    """Compute the shoulder-to-hip ratio at the most-stable frame.

    Args:
        landmarks: (T, 33, k) MediaPipe landmarks.
    Returns:
        Ratio of shoulder distance to hip distance (in image space). Returns
        1.0 if either denominator is too small to be reliable.
    """
    if landmarks.ndim != 3 or landmarks.shape[1] != 33:
        raise ValueError(f"expected (T, 33, k); got {landmarks.shape}")
    T = landmarks.shape[0]
    intensity = per_frame_velocity_magnitude(landmarks)
    # Trim the head and tail (first/last 10% of frames) to avoid pose-detection
    # warmup, then pick the lowest-velocity frame from the middle.
    trim = max(1, T // 10)
    candidates = intensity[trim : T - trim] if T > 2 * trim else intensity
    offset = trim if T > 2 * trim else 0
    idx = int(np.argmin(candidates)) + offset

    pts = landmarks[idx, :, :2].astype(np.float32)
    sho = np.linalg.norm(pts[MP_INDEX["left_shoulder"]] - pts[MP_INDEX["right_shoulder"]])
    hip = np.linalg.norm(pts[MP_INDEX["left_hip"]] - pts[MP_INDEX["right_hip"]])
    if sho < 1e-4 or hip < 1e-4:
        return 1.0
    return float(sho / hip)


def body_type_bucket(landmarks: np.ndarray) -> str:
    """Map a clip's shoulder/hip ratio to one of {narrow, balanced, broad}."""
    cfg = _body_type_config()
    thresholds = cfg["ratio_thresholds"]
    narrow_max = float(thresholds["narrow"])
    balanced_max = float(thresholds["balanced"])

    ratio = shoulder_hip_ratio(landmarks)
    if ratio < narrow_max:
        return "narrow"
    if ratio < balanced_max:
        return "balanced"
    return "broad"
