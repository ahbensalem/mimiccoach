"""Unit tests for rule-based coaching tip generation."""
from __future__ import annotations

import pytest

from pipeline.coach import coach_from_per_phase


def test_picks_lowest_scoring_phase() -> None:
    scores = [0.91, 0.84, 0.62, 0.78, 0.95]
    names = ["stance", "toss", "trophy", "contact", "follow_through"]
    tip = coach_from_per_phase("tennis_serve", scores, names)
    assert tip.weakest_phase == "trophy"
    assert tip.score == pytest.approx(0.62)
    assert "trophy" in tip.tip.lower() or "racket" in tip.tip.lower()


def test_emits_motion_specific_tip_when_known() -> None:
    scores = [0.5]
    names = ["descent"]
    tip = coach_from_per_phase("fitness_squat", scores, names)
    assert "Descend" in tip.tip or "hips" in tip.tip


def test_falls_back_when_motion_unknown() -> None:
    scores = [0.4, 0.6]
    names = ["foo", "bar"]
    tip = coach_from_per_phase("not_a_motion", scores, names)
    assert tip.weakest_phase == "foo"
    assert "session" in tip.tip.lower() or "compare" in tip.tip.lower()


def test_validates_input_lengths() -> None:
    with pytest.raises(ValueError):
        coach_from_per_phase("tennis_serve", [], [])
    with pytest.raises(ValueError):
        coach_from_per_phase("tennis_serve", [0.5, 0.6], ["only_one"])


def test_emits_for_every_supported_motion() -> None:
    """Every motion in motions.yaml should have at least one phase tip per phase."""
    from pipeline.segment import phase_names

    for motion in (
        "tennis_serve", "tennis_forehand", "tennis_backhand",
        "fitness_squat", "fitness_bench_press", "fitness_bent_over_row",
        "golf_swing",
    ):
        names = phase_names(motion)
        for i, n in enumerate(names):
            scores = [1.0] * len(names)
            scores[i] = 0.0  # force this phase to be the weakest
            tip = coach_from_per_phase(motion, scores, names)
            assert tip.weakest_phase == n
            # Must produce a non-empty, non-generic tip for known motion+phase combos.
            assert tip.tip
            assert tip.tip != "Focus your next session on this phase — record again and compare.", (
                f"motion={motion!r} phase={n!r} fell back to the generic tip"
            )
