"""Unit tests for body-type bucketing."""
from __future__ import annotations

import numpy as np

from pipeline.body_type import body_type_bucket, shoulder_hip_ratio
from pipeline.skeleton_map import MP_INDEX


def _pose_with_ratio(shoulder_w: float, hip_w: float, T: int = 30) -> np.ndarray:
    base = np.zeros((33, 4), dtype=np.float32)
    base[:, 3] = 0.9
    cx = 0.5
    base[MP_INDEX["left_shoulder"]] = [cx - shoulder_w / 2, 0.30, 0.0, 0.9]
    base[MP_INDEX["right_shoulder"]] = [cx + shoulder_w / 2, 0.30, 0.0, 0.9]
    base[MP_INDEX["left_hip"]] = [cx - hip_w / 2, 0.60, 0.0, 0.9]
    base[MP_INDEX["right_hip"]] = [cx + hip_w / 2, 0.60, 0.0, 0.9]
    base[MP_INDEX["nose"]] = [cx, 0.15, 0.0, 0.9]
    return np.tile(base, (T, 1, 1))


def test_ratio_balanced() -> None:
    clip = _pose_with_ratio(shoulder_w=0.20, hip_w=0.18)
    r = shoulder_hip_ratio(clip)
    assert abs(r - (0.20 / 0.18)) < 1e-3


def test_bucket_narrow() -> None:
    clip = _pose_with_ratio(shoulder_w=0.16, hip_w=0.18)  # ratio ≈ 0.89
    assert body_type_bucket(clip) == "narrow"


def test_bucket_balanced() -> None:
    clip = _pose_with_ratio(shoulder_w=0.22, hip_w=0.20)  # ratio = 1.10
    assert body_type_bucket(clip) == "balanced"


def test_bucket_broad() -> None:
    clip = _pose_with_ratio(shoulder_w=0.30, hip_w=0.22)  # ratio ≈ 1.36
    assert body_type_bucket(clip) == "broad"


def test_ratio_robust_to_zero_landmarks() -> None:
    clip = np.zeros((10, 33, 4), dtype=np.float32)
    # All zeros: shoulders and hips both collapse to a point.
    r = shoulder_hip_ratio(clip)
    assert r == 1.0
