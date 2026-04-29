"""Unit tests for MediaPipe-33 → H36M-17 mapping."""
from __future__ import annotations

import numpy as np

from pipeline.skeleton_map import (
    H36M_INDEX,
    H36M_JOINTS,
    MEDIAPIPE_LANDMARKS,
    MP_INDEX,
    mediapipe_to_h36m,
)


def test_lookup_tables_are_consistent() -> None:
    assert len(MEDIAPIPE_LANDMARKS) == 33
    assert len(H36M_JOINTS) == 17
    assert len(MP_INDEX) == 33
    assert len(H36M_INDEX) == 17
    # Round-trip: name → index → name
    for name, idx in MP_INDEX.items():
        assert MEDIAPIPE_LANDMARKS[idx] == name
    for name, idx in H36M_INDEX.items():
        assert H36M_JOINTS[idx] == name


def test_output_shape() -> None:
    mp = np.random.default_rng(0).standard_normal((50, 33, 3)).astype(np.float32)
    h = mediapipe_to_h36m(mp)
    assert h.shape == (50, 17, 3)
    assert h.dtype == np.float32


def test_passthrough_joints() -> None:
    """Joints that copy directly should be bit-identical."""
    mp = np.random.default_rng(0).standard_normal((10, 33, 3)).astype(np.float32)
    h = mediapipe_to_h36m(mp)
    pairs = [
        ("right_hip", "right_hip"),
        ("right_knee", "right_knee"),
        ("right_ankle", "right_ankle"),
        ("left_hip", "left_hip"),
        ("left_knee", "left_knee"),
        ("left_ankle", "left_ankle"),
        ("left_shoulder", "left_shoulder"),
        ("left_elbow", "left_elbow"),
        ("left_wrist", "left_wrist"),
        ("right_shoulder", "right_shoulder"),
        ("right_elbow", "right_elbow"),
        ("right_wrist", "right_wrist"),
        ("head", "nose"),
    ]
    for h_name, mp_name in pairs:
        np.testing.assert_array_equal(
            h[:, H36M_INDEX[h_name]],
            mp[:, MP_INDEX[mp_name]],
            err_msg=f"{h_name} should equal MediaPipe {mp_name}",
        )


def test_derived_pelvis_is_hip_midpoint() -> None:
    mp = np.zeros((1, 33, 3), dtype=np.float32)
    mp[0, MP_INDEX["left_hip"]] = [0.0, 1.0, 0.0]
    mp[0, MP_INDEX["right_hip"]] = [2.0, 3.0, 0.0]
    h = mediapipe_to_h36m(mp)
    np.testing.assert_array_almost_equal(
        h[0, H36M_INDEX["pelvis"]], [1.0, 2.0, 0.0]
    )


def test_derived_thorax_is_shoulder_midpoint() -> None:
    mp = np.zeros((1, 33, 3), dtype=np.float32)
    mp[0, MP_INDEX["left_shoulder"]] = [0.0, 0.0, 0.0]
    mp[0, MP_INDEX["right_shoulder"]] = [4.0, 4.0, 0.0]
    h = mediapipe_to_h36m(mp)
    np.testing.assert_array_almost_equal(
        h[0, H36M_INDEX["thorax"]], [2.0, 2.0, 0.0]
    )


def test_derived_spine_is_pelvis_thorax_midpoint() -> None:
    mp = np.zeros((1, 33, 3), dtype=np.float32)
    mp[0, MP_INDEX["left_hip"]] = [0.0, 0.0, 0.0]
    mp[0, MP_INDEX["right_hip"]] = [0.0, 0.0, 0.0]
    mp[0, MP_INDEX["left_shoulder"]] = [0.0, 4.0, 0.0]
    mp[0, MP_INDEX["right_shoulder"]] = [0.0, 4.0, 0.0]
    h = mediapipe_to_h36m(mp)
    np.testing.assert_array_almost_equal(
        h[0, H36M_INDEX["spine"]], [0.0, 2.0, 0.0]
    )


def test_visibility_channel_is_passed_through() -> None:
    mp = np.zeros((1, 33, 4), dtype=np.float32)
    mp[..., 3] = 0.7
    h = mediapipe_to_h36m(mp)
    assert h.shape == (1, 17, 4)
    # Passthrough joints keep visibility = 0.7
    assert h[0, H36M_INDEX["right_hip"], 3] == np.float32(0.7)
    # Derived joints average visibility
    assert h[0, H36M_INDEX["pelvis"], 3] == np.float32(0.7)


def test_rejects_wrong_shape() -> None:
    import pytest

    bad = np.zeros((10, 17, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        mediapipe_to_h36m(bad)
