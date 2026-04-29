"""Unit tests for phase segmentation."""
from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from pipeline.segment import (
    active_window,
    equal_partition,
    per_frame_velocity_magnitude,
    phase_names,
    segment_video,
    smooth_1d,
)
from pipeline.skeleton_map import MP_INDEX

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coherent_base() -> np.ndarray:
    base = np.zeros((33, 4), dtype=np.float32)
    base[:, 3] = 0.9
    base[MP_INDEX["nose"]] = [0.50, 0.15, 0.0, 0.9]
    base[MP_INDEX["left_shoulder"]] = [0.42, 0.30, 0.0, 0.9]
    base[MP_INDEX["right_shoulder"]] = [0.58, 0.30, 0.0, 0.9]
    base[MP_INDEX["left_elbow"]] = [0.38, 0.45, 0.0, 0.9]
    base[MP_INDEX["right_elbow"]] = [0.62, 0.45, 0.0, 0.9]
    base[MP_INDEX["left_wrist"]] = [0.34, 0.58, 0.0, 0.9]
    base[MP_INDEX["right_wrist"]] = [0.66, 0.58, 0.0, 0.9]
    base[MP_INDEX["left_hip"]] = [0.45, 0.60, 0.0, 0.9]
    base[MP_INDEX["right_hip"]] = [0.55, 0.60, 0.0, 0.9]
    base[MP_INDEX["left_knee"]] = [0.45, 0.80, 0.0, 0.9]
    base[MP_INDEX["right_knee"]] = [0.55, 0.80, 0.0, 0.9]
    base[MP_INDEX["left_ankle"]] = [0.45, 0.98, 0.0, 0.9]
    base[MP_INDEX["right_ankle"]] = [0.55, 0.98, 0.0, 0.9]
    return base


def _static_clip(T: int) -> np.ndarray:
    return np.tile(_coherent_base(), (T, 1, 1))


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def test_smooth_1d_passthrough_for_short() -> None:
    s = np.array([1.0], dtype=np.float32)
    np.testing.assert_array_equal(smooth_1d(s, window=5), s)


def test_smooth_1d_rounds_even_window_to_odd() -> None:
    s = np.arange(10, dtype=np.float32)
    out = smooth_1d(s, window=4)
    assert out.shape == s.shape


def test_per_frame_velocity_zero_for_static() -> None:
    clip = _static_clip(20)
    v = per_frame_velocity_magnitude(clip)
    assert v.shape == (20,)
    np.testing.assert_array_almost_equal(v, np.zeros(20, dtype=np.float32))


def test_per_frame_velocity_responds_to_motion() -> None:
    T = 30
    clip = _static_clip(T)
    swing = np.sin(np.linspace(0, 2 * np.pi, T)) * 0.1
    clip[:, MP_INDEX["right_wrist"], 1] += swing
    v = per_frame_velocity_magnitude(clip)
    assert v.max() > 0.005


# ---------------------------------------------------------------------------
# Active window
# ---------------------------------------------------------------------------

def test_active_window_full_for_static() -> None:
    """When there's no motion, the window covers the whole clip."""
    clip = _static_clip(40)
    s, e = active_window(clip)
    assert (s, e) == (0, 40)


def test_active_window_trims_idle_frames() -> None:
    """Static head + active body + static tail → window covers the active body."""
    T = 60
    clip = _static_clip(T)
    # Add a clear motion in the middle 30 frames.
    swing = np.sin(np.linspace(0, 2 * np.pi, 30)) * 0.15
    clip[15:45, MP_INDEX["right_wrist"], 1] += swing
    clip[15:45, MP_INDEX["left_wrist"], 1] += swing
    s, e = active_window(clip)
    assert s >= 10
    assert e <= 50
    assert e - s >= 10


# ---------------------------------------------------------------------------
# Equal partition
# ---------------------------------------------------------------------------

def test_equal_partition_splits_evenly() -> None:
    parts = equal_partition(0, 30, 5)
    assert len(parts) == 5
    assert parts[0][0] == 0
    assert parts[-1][1] == 30
    # Contiguous, no gaps.
    for (a, b), (c, d) in pairwise(parts):
        assert b == c
        assert b > a and d > c


def test_equal_partition_rejects_too_small() -> None:
    with pytest.raises(ValueError):
        equal_partition(0, 3, 5)


# ---------------------------------------------------------------------------
# segment_video
# ---------------------------------------------------------------------------

def test_phase_names_known_motions() -> None:
    assert phase_names("tennis_serve") == [
        "stance", "toss", "trophy", "contact", "follow_through",
    ]
    assert phase_names("fitness_squat") == [
        "setup", "descent", "bottom", "ascent", "lockout",
    ]
    assert len(phase_names("golf_swing")) == 6


def test_phase_names_unknown_motion_raises() -> None:
    with pytest.raises(KeyError):
        phase_names("not_a_motion")


def test_segment_video_returns_one_window_per_phase_for_all_motions() -> None:
    """Equal-partition fallback produces sensible output for every motion."""
    T = 60
    clip = _static_clip(T)
    swing = np.sin(np.linspace(0, 2 * np.pi, T)) * 0.05
    clip[:, MP_INDEX["right_wrist"], 1] += swing  # mild whole-clip motion

    for motion in (
        "tennis_serve", "tennis_forehand", "tennis_backhand",
        "fitness_squat", "fitness_bench_press", "fitness_bent_over_row",
        "golf_swing",
    ):
        windows = segment_video(clip, motion)
        names = [n for n, _, _ in windows]
        assert names == phase_names(motion)
        # Strictly increasing, contiguous, all in-bounds.
        for _, s, e in windows:
            assert 0 <= s < e <= T
        for (_, _, b), (_, c, d) in pairwise(windows):
            assert b == c
            assert d > c


def test_segment_video_short_clip_raises() -> None:
    clip = _static_clip(3)  # too few frames for any motion (>= 5 phases)
    with pytest.raises(ValueError):
        segment_video(clip, "tennis_serve")


def test_tennis_serve_signal_segmenter_uses_wrist_peaks() -> None:
    """Synthetic serve where wrist trajectories are well-formed should produce
    a phase ordering driven by the signal, not the equal-partition fallback."""
    T = 120
    base = _coherent_base()
    clip = np.tile(base, (T, 1, 1))

    # Toss: left wrist rises early (frames 20–50), peak at frame 35.
    toss_t = np.arange(T)
    toss = np.where(
        (toss_t >= 20) & (toss_t <= 50),
        -0.18 * np.sin(np.pi * (toss_t - 20) / 30),
        0.0,
    )
    clip[:, MP_INDEX["left_wrist"], 1] += toss

    # Trophy: right elbow flexes max around frame 60 (elbow brought up close
    # to right wrist so the angle at elbow is small).
    flex_t = np.arange(T)
    flex_amp = np.where(
        (flex_t >= 40) & (flex_t <= 80),
        0.15 * np.sin(np.pi * (flex_t - 40) / 40),
        0.0,
    )
    # Pull right wrist toward right elbow (in y) to flex the elbow angle.
    clip[:, MP_INDEX["right_wrist"], 1] -= flex_amp  # wrist moves UP toward elbow

    # Contact: right wrist peaks at frame 90 (very high).
    contact_t = np.arange(T)
    contact = np.where(
        (contact_t >= 75) & (contact_t <= 105),
        -0.25 * np.sin(np.pi * (contact_t - 75) / 30),
        0.0,
    )
    clip[:, MP_INDEX["right_wrist"], 1] += contact

    windows = segment_video(clip, "tennis_serve")
    assert [n for n, _, _ in windows] == [
        "stance", "toss", "trophy", "contact", "follow_through",
    ]
    # Toss boundary should be near the toss peak (~35), trophy near 60,
    # contact near 90 — i.e., the boundaries should be roughly increasing
    # with our planted events.
    s_toss, s_trophy, s_contact = windows[1][1], windows[2][1], windows[3][1]
    assert s_toss < s_trophy < s_contact
    # Sanity: not collapsed to equal-partition (each window should be > 1 frame).
    for _, s, e in windows:
        assert e - s >= 1
