"""MediaPipe Pose (33 landmarks) ↔ Human3.6M (17 joints) skeleton mapping.

MediaPipe Pose Landmarker emits 33 landmarks per frame; MotionBERT and most
academic pose models expect Human3.6M-17 ordering. This module provides a
pure-numpy mapping between the two, including derived joints (pelvis, spine,
thorax) that MediaPipe does not emit directly.

The H36M-17 ordering used here matches MotionBERT's input convention.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe Pose Landmarker — 33 landmarks
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
# ---------------------------------------------------------------------------
MEDIAPIPE_LANDMARKS: tuple[str, ...] = (
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)
MP_INDEX: Mapping[str, int] = {name: i for i, name in enumerate(MEDIAPIPE_LANDMARKS)}

# ---------------------------------------------------------------------------
# Human3.6M — 17 joints (MotionBERT input convention)
# ---------------------------------------------------------------------------
H36M_JOINTS: tuple[str, ...] = (
    "pelvis",         # 0  midpoint(L_hip, R_hip)
    "right_hip",      # 1
    "right_knee",     # 2
    "right_ankle",    # 3
    "left_hip",       # 4
    "left_knee",      # 5
    "left_ankle",     # 6
    "spine",          # 7  midpoint(pelvis, thorax)
    "thorax",         # 8  midpoint(L_shoulder, R_shoulder)
    "neck",           # 9  midpoint(thorax, nose)
    "head",           # 10 nose (top-of-head proxy)
    "left_shoulder",  # 11
    "left_elbow",     # 12
    "left_wrist",     # 13
    "right_shoulder", # 14
    "right_elbow",    # 15
    "right_wrist",    # 16
)
H36M_INDEX: Mapping[str, int] = {name: i for i, name in enumerate(H36M_JOINTS)}


def mediapipe_to_h36m(mp_landmarks: np.ndarray) -> np.ndarray:
    """Map MediaPipe-33 landmarks to Human3.6M-17 joints.

    Args:
        mp_landmarks: shape (T, 33, k) where T is the number of frames and
            k is the number of channels per landmark. Typically k=3 (x,y,z)
            or k=4 (x,y,z,visibility).

    Returns:
        shape (T, 17, k). Derived joints (pelvis, spine, thorax, neck) are
        midpoints of the corresponding source joints; visibility (if present)
        is averaged.
    """
    if mp_landmarks.ndim != 3 or mp_landmarks.shape[1] != 33:
        raise ValueError(f"expected (T, 33, k); got {mp_landmarks.shape}")

    L_HIP = MP_INDEX["left_hip"]
    R_HIP = MP_INDEX["right_hip"]
    L_KNEE = MP_INDEX["left_knee"]
    R_KNEE = MP_INDEX["right_knee"]
    L_ANK = MP_INDEX["left_ankle"]
    R_ANK = MP_INDEX["right_ankle"]
    L_SHO = MP_INDEX["left_shoulder"]
    R_SHO = MP_INDEX["right_shoulder"]
    L_ELB = MP_INDEX["left_elbow"]
    R_ELB = MP_INDEX["right_elbow"]
    L_WRI = MP_INDEX["left_wrist"]
    R_WRI = MP_INDEX["right_wrist"]
    NOSE = MP_INDEX["nose"]

    pelvis = (mp_landmarks[:, L_HIP] + mp_landmarks[:, R_HIP]) * 0.5
    thorax = (mp_landmarks[:, L_SHO] + mp_landmarks[:, R_SHO]) * 0.5
    spine = (pelvis + thorax) * 0.5
    neck = (thorax + mp_landmarks[:, NOSE]) * 0.5

    h36m = np.stack(
        [
            pelvis,                       # 0
            mp_landmarks[:, R_HIP],       # 1
            mp_landmarks[:, R_KNEE],      # 2
            mp_landmarks[:, R_ANK],       # 3
            mp_landmarks[:, L_HIP],       # 4
            mp_landmarks[:, L_KNEE],      # 5
            mp_landmarks[:, L_ANK],       # 6
            spine,                        # 7
            thorax,                       # 8
            neck,                         # 9
            mp_landmarks[:, NOSE],        # 10
            mp_landmarks[:, L_SHO],       # 11
            mp_landmarks[:, L_ELB],       # 12
            mp_landmarks[:, L_WRI],       # 13
            mp_landmarks[:, R_SHO],       # 14
            mp_landmarks[:, R_ELB],       # 15
            mp_landmarks[:, R_WRI],       # 16
        ],
        axis=1,
    )
    return h36m
