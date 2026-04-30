"""Tests for the GolfDB loader's offline logic.

The full ``iter_rows`` integration test would need a MediaPipe model and
a real cropped GolfDB clip on disk; those aren't checked into the repo
(``data/`` is gitignored). These tests cover the deterministic pieces
that don't depend on network or native libs:

  * 8-event → 6-phase mapping math
  * ID offset that keeps golfdb ids disjoint from synthetic ids
  * pose-cache save/load round-trip used by the live overlay
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reference import loader_golfdb
from reference.loader_golfdb import (
    ID_OFFSET,
    PHASE_NAMES,
    _phase_boundaries_from_events,
    _save_pose_cache,
    landmarks_for_entry,
)
from reference.synthetic import generate as generate_synthetic


def test_phase_boundaries_canonical_mapping() -> None:
    # Events use the GolfDB convention: 10 frame indices in source video,
    # bracketed by start/end padding. Cropped clip frame 0 == events[0].
    events = (10, 20, 25, 35, 45, 55, 65, 70, 75, 90)
    boundaries = _phase_boundaries_from_events(events, num_frames=81)
    assert [name for name, _, _ in boundaries] == list(PHASE_NAMES)

    # Phases collapse the 8 GT events into 6 windows the way our
    # motions.yaml golf_swing schema expects.
    expected = [
        ("address", 0, 10),    # pad-start → Address
        ("backswing", 10, 25), # Address    → Mid-Backswing
        ("top", 25, 35),       # Mid-Back   → Top
        ("downswing", 35, 45), # Top        → Mid-Down
        ("impact", 45, 55),    # Mid-Down   → Impact
        ("finish", 55, 80),    # Impact     → end-pad (clamped to num_frames)
    ]
    assert boundaries == expected


def test_phase_boundaries_falls_back_to_equal_partition_on_degenerate_event() -> None:
    # Some GolfDB annotations have collapsed events (e.g. mid_back == top).
    # The mapper should fall back to equal-partition rather than emit a
    # zero-width phase that phase_token() would reject.
    events = (0, 5, 10, 20, 20, 30, 40, 45, 50, 60)  # top == mid_back
    boundaries = _phase_boundaries_from_events(events, num_frames=60)
    assert len(boundaries) == len(PHASE_NAMES)
    for _name, s, e in boundaries:
        assert e > s, "every phase must have positive width"
    # Equal partition is contiguous and covers the whole clip.
    assert boundaries[0][1] == 0
    assert boundaries[-1][2] == 60


def test_phase_boundaries_rejects_too_short_clip() -> None:
    events = (0, 1, 1, 1, 1, 1, 1, 1, 1, 4)  # forces fallback, only 4 frames
    with pytest.raises(RuntimeError, match="too short"):
        _phase_boundaries_from_events(events, num_frames=4)


def test_id_offset_disjoint_from_synthetic() -> None:
    # The synthetic library currently emits ids 1..~250; the offset has to
    # stay clear of that even when synthetic grows.
    synthetic_max = max(e.id for e in generate_synthetic())
    assert ID_OFFSET > synthetic_max * 10, (
        "ID_OFFSET should leave generous headroom over synthetic ids"
    )


def test_landmarks_for_entry_below_offset_returns_none(tmp_path: Path) -> None:
    # Synthetic ids must not accidentally hit the golfdb cache path.
    assert landmarks_for_entry(1, data_root=tmp_path) is None
    assert landmarks_for_entry(ID_OFFSET - 1, data_root=tmp_path) is None


def test_landmarks_for_entry_missing_cache_returns_none(tmp_path: Path) -> None:
    assert landmarks_for_entry(ID_OFFSET + 1, data_root=tmp_path) is None


def test_pose_cache_roundtrip(tmp_path: Path) -> None:
    # _save_pose_cache + landmarks_for_entry have to round-trip the
    # landmarks exactly so the live overlay's pro skeleton matches what
    # the loader saw at embed time.
    landmarks = np.random.default_rng(42).standard_normal((20, 33, 4)).astype(np.float32)
    entry_id = ID_OFFSET + 7

    _save_pose_cache(tmp_path, entry_id, landmarks, fps=30.0, motion="golf_swing")
    # Bypass lru_cache so a fresh load actually hits disk.
    loader_golfdb._load_pose_cache.cache_clear()

    info = landmarks_for_entry(entry_id, data_root=tmp_path)
    assert info is not None
    pose, motion, fps = info
    assert pose.shape == (20, 33, 4)
    assert motion == "golf_swing"
    assert fps == 30.0
    np.testing.assert_array_equal(pose, landmarks)
