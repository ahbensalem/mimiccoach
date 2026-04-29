"""Verifies the demo's 'filter swing' moment: changing skill_level or
body_type at query time should visibly change the top match.

This is the beat we lean on in the architecture-explainer + filter-demo
sections of the demo video — toggling a chip in the UI returns a
different pro, all powered by Qdrant's payload-indexed filters running
alongside the multivector MaxSim query.
"""
from __future__ import annotations

from collections.abc import Iterator

import pytest

import app
from qdrant_io.client import make_client
from qdrant_io.schema import create_collection
from qdrant_io.upsert import manifest_to_points, upsert_points
from reference.synthetic import generate as generate_synthetic
from tests.test_all_motions_e2e import _user_clip_for_motion  # type: ignore[import-not-found]


@pytest.fixture(scope="module")
def seeded_client(tmp_path_factory) -> Iterator:
    from os import environ

    qdrant_dir = tmp_path_factory.mktemp("qdrant_filter")
    environ["QDRANT_PATH"] = str(qdrant_dir)
    environ.pop("QDRANT_URL", None)

    client = make_client()
    create_collection(client, recreate=True)
    rows = [
        {"id": e.id, "phase_tokens": e.phase_tokens, "payload": e.payload}
        for e in generate_synthetic()
    ]
    upsert_points(client, manifest_to_points(rows))
    yield client
    client.close()
    environ.pop("QDRANT_PATH", None)


def test_skill_level_filter_changes_top_match(seeded_client) -> None:
    landmarks = _user_clip_for_motion("tennis_serve")
    pro = app.analyze_from_landmarks(
        landmarks, motion="tennis_serve", skill_level="pro", client=seeded_client,
    )
    beg = app.analyze_from_landmarks(
        landmarks, motion="tennis_serve", skill_level="beginner", client=seeded_client,
    )

    assert pro["match"] is not None
    assert beg["match"] is not None
    assert pro["match"]["skill_level"] == "pro"
    assert beg["match"]["skill_level"] == "beginner"
    # Changing skill_level should pick a different point — that's the demo beat.
    assert pro["match"]["point_id"] != beg["match"]["point_id"]


def test_body_type_filter_changes_top_match(seeded_client) -> None:
    landmarks = _user_clip_for_motion("fitness_squat")
    narrow = app.analyze_from_landmarks(
        landmarks, motion="fitness_squat", body_type_override="narrow", client=seeded_client,
    )
    broad = app.analyze_from_landmarks(
        landmarks, motion="fitness_squat", body_type_override="broad", client=seeded_client,
    )

    assert narrow["match"] is not None
    assert broad["match"] is not None
    assert narrow["match"]["body_type"] == "narrow"
    assert broad["match"]["body_type"] == "broad"
    assert narrow["match"]["point_id"] != broad["match"]["point_id"]


def test_no_filter_returns_global_top_match(seeded_client) -> None:
    """Without filters, retrieval picks the closest pro across all
    skill_levels and body_types — this is the default UX."""
    landmarks = _user_clip_for_motion("golf_swing")
    result = app.analyze_from_landmarks(
        landmarks, motion="golf_swing", client=seeded_client,
    )
    assert result["match"] is not None
    assert result["match"]["score"] > 0.30
    assert result["filters_applied"]["skill_level"] is None
    assert result["filters_applied"]["body_type"] is None
