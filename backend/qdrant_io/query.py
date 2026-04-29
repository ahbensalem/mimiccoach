"""Multivector + payload-filter query against the `motions` collection.

This is the project's headline Qdrant moment. The user's pose tokens
(one per phase) are sent as a multivector query; the server computes
late-interaction MaxSim against each stored point's phase tokens,
optionally constrained by sport / motion / skill_level / body_type
payload filters.

We also recompute the *per-phase* MaxSim breakdown client-side from
the matched stored tokens — this is what drives the UI's per-phase
score chips and the rule-based coaching tip on the weakest phase.
The aggregate Qdrant score and our recomputed per-phase scores agree
by construction (MaxSim = mean over query tokens of per-token max-sim).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from .schema import COLLECTION_NAME, PHASE_TOKENS_FIELD


@dataclass
class MotionMatch:
    point_id: int | str
    score: float
    """Aggregate MaxSim score from Qdrant."""
    per_phase_scores: list[float]
    """Per-query-token max-sim against the matched stored tokens (length =
    number of query phases)."""
    payload: dict[str, Any]


def build_filter(
    *,
    sport: str | None = None,
    motion: str | None = None,
    skill_level: str | None = None,
    body_type: str | None = None,
) -> Filter | None:
    must = []
    for key, val in (
        ("sport", sport),
        ("motion", motion),
        ("skill_level", skill_level),
        ("body_type", body_type),
    ):
        if val is not None:
            must.append(FieldCondition(key=key, match=MatchValue(value=val)))
    return Filter(must=must) if must else None


def query_motions(
    client: QdrantClient,
    phase_tokens: list[list[float]] | np.ndarray,
    *,
    sport: str | None = None,
    motion: str | None = None,
    skill_level: str | None = None,
    body_type: str | None = None,
    limit: int = 5,
) -> list[MotionMatch]:
    """Run a multivector + payload-filter query and return top-`limit` matches
    with per-phase score breakdown."""
    query_arr = np.asarray(phase_tokens, dtype=np.float32)
    if query_arr.ndim != 2:
        raise ValueError(f"phase_tokens must be 2D; got shape {query_arr.shape}")

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_arr.tolist(),
        using=PHASE_TOKENS_FIELD,
        query_filter=build_filter(
            sport=sport,
            motion=motion,
            skill_level=skill_level,
            body_type=body_type,
        ),
        limit=limit,
        with_payload=True,
        with_vectors=True,
    )

    matches: list[MotionMatch] = []
    for point in response.points:
        stored_vec = point.vector
        if isinstance(stored_vec, dict):
            stored_vec = stored_vec.get(PHASE_TOKENS_FIELD)
        if stored_vec is None:
            # Server didn't return the stored tokens; emit an empty per-phase array.
            per_phase = [float(point.score)] * query_arr.shape[0]
        else:
            stored = np.asarray(stored_vec, dtype=np.float32)  # (n_stored, 512)
            sims = query_arr @ stored.T  # cosine since both L2-normalized
            per_phase = sims.max(axis=1).astype(float).tolist()

        matches.append(
            MotionMatch(
                point_id=point.id,
                score=float(point.score),
                per_phase_scores=per_phase,
                payload=dict(point.payload or {}),
            )
        )
    return matches
