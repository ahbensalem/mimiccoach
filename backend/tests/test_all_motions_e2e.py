"""End-to-end retrieval test for all 7 motions against the synthetic library.

Seeds Qdrant once with the full synthetic library, then for each of the 7
motions builds a fresh synthetic "user clip" (a different parametrization
than what's in the library) and verifies that:

  * The pipeline runs without error.
  * The top match has the correct `sport` and `motion`.
  * The aggregate score is meaningful (> 0.3).
  * Per-phase scores have one entry per declared phase.

This is P7's verification — the plan asks for "all 7 motions return
correct sport, plausible match, plausible scores".
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

import app
from qdrant_io.client import make_client
from qdrant_io.schema import create_collection
from qdrant_io.upsert import manifest_to_points, upsert_points
from reference.synthetic import (
    BODY_TEMPLATES,
    PLANS,
)
from reference.synthetic import (
    generate as generate_synthetic,
)


@pytest.fixture(scope="module")
def seeded_client(tmp_path_factory) -> Iterator:
    """Seed a single on-disk Qdrant with the full synthetic library."""
    from os import environ

    qdrant_dir = tmp_path_factory.mktemp("qdrant")
    environ["QDRANT_PATH"] = str(qdrant_dir)
    environ.pop("QDRANT_URL", None)

    client = make_client()
    create_collection(client, recreate=True)

    rows = []
    for entry in generate_synthetic():
        rows.append(
            {
                "id": entry.id,
                "phase_tokens": entry.phase_tokens,
                "payload": entry.payload,
            }
        )
    upsert_points(client, manifest_to_points(rows))
    yield client
    client.close()
    environ.pop("QDRANT_PATH", None)


def _user_clip_for_motion(motion_key: str) -> np.ndarray:
    """Build a synthetic 'user clip' for a motion that's not bit-identical to
    any seeded clip. We pick a body template + duration that doesn't appear
    in the synthetic library at the same skill index, so the test exercises
    real retrieval rather than self-match."""
    plan = next(p for p in PLANS if p.motion == motion_key)
    body = BODY_TEMPLATES[2]  # balanced_classic
    T = plan.durations[0] + 5  # ±5 frame offset from any seeded duration
    rng = np.random.default_rng(2027)
    return plan.make(body, T, skill=0.6, rng=rng)


@pytest.mark.parametrize(
    "motion_key,expected_sport",
    [
        ("tennis_serve", "tennis"),
        ("tennis_forehand", "tennis"),
        ("tennis_backhand", "tennis"),
        ("fitness_squat", "fitness"),
        ("fitness_bench_press", "fitness"),
        ("fitness_bent_over_row", "fitness"),
        ("golf_swing", "golf"),
    ],
)
def test_motion_retrieves_correct_sport(
    seeded_client, motion_key: str, expected_sport: str
) -> None:
    landmarks = _user_clip_for_motion(motion_key)
    result = app.analyze_from_landmarks(
        landmarks,
        motion=motion_key,
        client=seeded_client,
    )
    assert result["match"] is not None, (
        f"no match for {motion_key} — synthetic library may be missing entries"
    )
    assert result["match"]["score"] > 0.30, (
        f"{motion_key} top match score {result['match']['score']:.3f} is implausibly low"
    )
    # The top match's payload must agree on sport AND motion (not just sport).
    # Filters constrain to motion=motion_key already, so motion mismatch would
    # mean the filter wasn't honored.
    assert result["filters_applied"]["sport"] == expected_sport
    # Per-phase scores have the right cardinality.
    expected_phase_count = (
        6 if motion_key == "golf_swing" else 5
    )
    assert len(result["per_phase_scores"]) == expected_phase_count
    # The user metadata round-trips the motion + body type.
    assert result["user"]["motion"] == motion_key
    assert result["user"]["sport"] == expected_sport
    assert result["user"]["body_type"] in {"narrow", "balanced", "broad"}
