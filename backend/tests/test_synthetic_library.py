"""Tests for the synthetic reference library generator."""
from __future__ import annotations

from collections import Counter

import numpy as np

from pipeline.embed import EMBED_DIM
from pipeline.segment import phase_names
from reference.synthetic import BODY_TEMPLATES, PLANS, generate


def test_generate_yields_expected_count() -> None:
    """7 motions × 6 body templates × N durations per motion."""
    expected = sum(len(BODY_TEMPLATES) * len(p.durations) for p in PLANS)
    rows = list(generate())
    assert len(rows) == expected


def test_generate_covers_all_seven_motions() -> None:
    rows = list(generate())
    motions = {r.payload["motion"] for r in rows}
    assert motions == {p.motion for p in PLANS}
    assert len(motions) == 7


def test_generate_produces_balanced_skill_distribution() -> None:
    rows = list(generate())
    counts = Counter(r.payload["skill_level"] for r in rows)
    # All three labels should be represented per the synthetic skill schedule.
    assert set(counts) == {"beginner", "intermediate", "pro"}
    for label, n in counts.items():
        assert n > 0, label


def test_generate_covers_all_three_body_types() -> None:
    rows = list(generate())
    bts = {r.payload["body_type"] for r in rows}
    assert bts == {"narrow", "balanced", "broad"}


def test_phase_token_shape_is_consistent() -> None:
    for r in generate():
        tokens = np.asarray(r.phase_tokens, dtype=np.float32)
        assert tokens.ndim == 2
        assert tokens.shape[1] == EMBED_DIM
        assert tokens.shape[0] == len(phase_names(r.payload["motion"]))
        # L2-normalized per phase.
        norms = np.linalg.norm(tokens, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones_like(norms), decimal=5)


def test_generate_is_deterministic() -> None:
    a = list(generate(seed=42))
    b = list(generate(seed=42))
    assert len(a) == len(b)
    for ra, rb in zip(a, b, strict=True):
        assert ra.id == rb.id
        assert ra.payload == rb.payload
        np.testing.assert_array_almost_equal(
            np.asarray(ra.phase_tokens), np.asarray(rb.phase_tokens), decimal=6
        )


def test_payloads_have_required_fields() -> None:
    required = {"sport", "motion", "skill_level", "body_type", "athlete",
                "source", "source_url", "license_note"}
    for r in generate():
        assert required <= r.payload.keys(), r.payload


def test_unique_ids() -> None:
    rows = list(generate())
    ids = [r.id for r in rows]
    assert len(set(ids)) == len(ids)
