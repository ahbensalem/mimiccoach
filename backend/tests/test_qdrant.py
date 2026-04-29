"""End-to-end test of the Qdrant integration: schema → upsert → query.

Uses qdrant-client's local mode (:memory: or path-based on-disk), so the
suite needs no external Qdrant instance. The real Qdrant Cloud
deployment is exercised separately via the `modal run` smoke test (P5).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qdrant_io.client import make_client
from qdrant_io.query import build_filter, query_motions
from qdrant_io.schema import (
    COLLECTION_NAME,
    EMBED_DIM,
    create_collection,
)
from qdrant_io.upsert import iter_manifest, manifest_to_points, upsert_manifest


def _unit(rng: np.random.Generator, dim: int = EMBED_DIM) -> list[float]:
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-8
    return v.tolist()


def _make_clip_tokens(rng: np.random.Generator, n_phases: int = 5) -> list[list[float]]:
    return [_unit(rng) for _ in range(n_phases)]


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Force the on-disk embedded mode so multivector is supported. The pure
    # in-memory mode in qdrant-client may lag on multivector features.
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    c = make_client()
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_create_collection_idempotent(client) -> None:
    create_collection(client)
    assert client.collection_exists(COLLECTION_NAME)

    # Second call without recreate is a no-op (no exception).
    create_collection(client)
    assert client.collection_exists(COLLECTION_NAME)


def test_create_collection_recreate_drops_existing(client) -> None:
    create_collection(client)
    rng = np.random.default_rng(0)
    points = manifest_to_points([
        {
            "id": 1,
            "phase_tokens": _make_clip_tokens(rng),
            "payload": {"sport": "tennis", "motion": "serve",
                        "skill_level": "pro", "body_type": "balanced"},
        }
    ])
    from qdrant_io.upsert import upsert_points
    upsert_points(client, points)

    create_collection(client, recreate=True)
    info = client.get_collection(COLLECTION_NAME)
    assert getattr(info, "points_count", 0) in (0, None)


def test_payload_indexes_create_without_error(client) -> None:
    """Local qdrant-client mode emits a UserWarning ('Payload indexes have no
    effect in the local Qdrant') and doesn't surface them in payload_schema.
    On production Qdrant Cloud the indexes are honored; we verify here that
    the create_payload_index calls are emitted without error. Behavior under
    filtering is exercised by `test_query_filter_excludes_other_sports`."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        create_collection(client, recreate=True)
    assert client.collection_exists(COLLECTION_NAME)


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def test_iter_manifest_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps({"a": 1}) + "\n\n" + json.dumps({"a": 2}) + "\n")
    rows = list(iter_manifest(p))
    assert rows == [{"a": 1}, {"a": 2}]


def test_manifest_to_points_validates_keys() -> None:
    with pytest.raises(ValueError):
        manifest_to_points([{"id": 1, "phase_tokens": [[0.0]]}])  # missing payload


def test_upsert_manifest_round_trip(client, tmp_path: Path) -> None:
    create_collection(client, recreate=True)
    rng = np.random.default_rng(0)

    entries = [
        {
            "id": i,
            "phase_tokens": _make_clip_tokens(rng),
            "payload": {
                "sport": "tennis",
                "motion": "serve",
                "skill_level": "pro" if i % 2 == 0 else "intermediate",
                "body_type": "balanced",
                "athlete": f"player_{i}",
            },
        }
        for i in range(1, 6)
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    n = upsert_manifest(client, manifest, batch_size=2)
    assert n == 5
    info = client.get_collection(COLLECTION_NAME)
    # qdrant-client returns either points_count or vectors_count depending on backend.
    assert (getattr(info, "points_count", None) or 0) >= 5


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def test_build_filter_returns_none_when_empty() -> None:
    assert build_filter() is None


def test_build_filter_combines_must_clauses() -> None:
    f = build_filter(sport="tennis", skill_level="pro")
    assert f is not None
    assert len(f.must) == 2


def test_query_motions_returns_top_k_with_per_phase_scores(client) -> None:
    create_collection(client, recreate=True)
    rng = np.random.default_rng(42)

    # Anchor clip we will later try to retrieve.
    anchor_tokens = _make_clip_tokens(rng, n_phases=5)
    entries = [
        {
            "id": 99,
            "phase_tokens": anchor_tokens,
            "payload": {"sport": "tennis", "motion": "serve",
                        "skill_level": "pro", "body_type": "balanced"},
        }
    ]
    # Plus some distractors with a different sport.
    for i in range(1, 5):
        entries.append({
            "id": i,
            "phase_tokens": _make_clip_tokens(rng, n_phases=5),
            "payload": {"sport": "fitness", "motion": "squat",
                        "skill_level": "pro", "body_type": "balanced"},
        })

    from qdrant_io.upsert import upsert_points
    upsert_points(client, manifest_to_points(entries))

    matches = query_motions(client, anchor_tokens, sport="tennis", limit=3)
    assert len(matches) >= 1
    # The anchor itself should be top-1 with near-perfect score.
    assert matches[0].point_id == 99
    assert matches[0].score > 0.99
    # Per-phase scores have one entry per query phase.
    assert len(matches[0].per_phase_scores) == 5
    for s in matches[0].per_phase_scores:
        assert 0.99 <= s <= 1.0001


def test_query_filter_excludes_other_sports(client) -> None:
    create_collection(client, recreate=True)
    rng = np.random.default_rng(7)
    anchor_tokens = _make_clip_tokens(rng, n_phases=5)

    from qdrant_io.upsert import upsert_points
    upsert_points(client, manifest_to_points([
        {
            "id": 1,
            "phase_tokens": anchor_tokens,
            "payload": {"sport": "tennis", "motion": "serve",
                        "skill_level": "pro", "body_type": "balanced"},
        },
        {
            "id": 2,
            "phase_tokens": anchor_tokens,  # identical vectors!
            "payload": {"sport": "fitness", "motion": "squat",
                        "skill_level": "pro", "body_type": "balanced"},
        },
    ]))

    # Without filter: both match.
    all_matches = query_motions(client, anchor_tokens, limit=5)
    sports = {m.payload["sport"] for m in all_matches}
    assert sports == {"tennis", "fitness"}

    # With sport=tennis filter: only the tennis point.
    only_tennis = query_motions(client, anchor_tokens, sport="tennis", limit=5)
    assert len(only_tennis) == 1
    assert only_tennis[0].payload["sport"] == "tennis"


def test_query_handles_skill_level_and_body_type_filters(client) -> None:
    create_collection(client, recreate=True)
    rng = np.random.default_rng(11)
    anchor = _make_clip_tokens(rng, n_phases=5)
    from qdrant_io.upsert import upsert_points
    upsert_points(client, manifest_to_points([
        {
            "id": 1,
            "phase_tokens": anchor,
            "payload": {"sport": "tennis", "motion": "serve",
                        "skill_level": "pro", "body_type": "narrow"},
        },
        {
            "id": 2,
            "phase_tokens": anchor,
            "payload": {"sport": "tennis", "motion": "serve",
                        "skill_level": "beginner", "body_type": "broad"},
        },
    ]))

    pro_matches = query_motions(client, anchor, sport="tennis", skill_level="pro")
    assert [m.payload["skill_level"] for m in pro_matches] == ["pro"]

    broad_matches = query_motions(client, anchor, sport="tennis", body_type="broad")
    assert [m.payload["body_type"] for m in broad_matches] == ["broad"]
