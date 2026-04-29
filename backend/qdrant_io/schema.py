"""Qdrant collection schema for MimicCoach.

Single collection `motions` with one multivector field `phase_tokens`.
Each point's `phase_tokens` is a list of 512-d L2-normalized vectors —
one per phase of the motion (5 or 6 per clip). Cosine distance with
MaxSim late-interaction comparator is the late-interaction primitive
that gives us per-phase retrieval; ColBERT/ColPali use the same pattern
for documents.

`HnswConfigDiff(m=0)` disables HNSW graph construction on the multivector
field — the Qdrant docs recommend this because multivector queries are
typically rerank-style and don't benefit from a graph index.

Payload indexes on sport / motion / skill_level / body_type let the
filter combinations in `qdrant_io.query.query_motions` run unchanged
alongside the multivector search.
"""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    MultiVectorComparator,
    MultiVectorConfig,
    PayloadSchemaType,
    VectorParams,
)

COLLECTION_NAME: str = "motions"
EMBED_DIM: int = 512
PHASE_TOKENS_FIELD: str = "phase_tokens"

PAYLOAD_KEYWORD_FIELDS: tuple[str, ...] = (
    "sport",
    "motion",
    "skill_level",
    "body_type",
)


def create_collection(client: QdrantClient, *, recreate: bool = False) -> None:
    """Create the `motions` collection if it doesn't exist (or recreate it)."""
    if client.collection_exists(COLLECTION_NAME):
        if not recreate:
            return
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            PHASE_TOKENS_FIELD: VectorParams(
                size=EMBED_DIM,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=HnswConfigDiff(m=0),
            ),
        },
    )

    for field in PAYLOAD_KEYWORD_FIELDS:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )


def collection_info(client: QdrantClient) -> dict[str, object]:
    """Compact summary used for diagnostics / smoke tests."""
    info = client.get_collection(COLLECTION_NAME)
    return {
        "exists": True,
        "vectors_count": getattr(info, "vectors_count", None),
        "points_count": getattr(info, "points_count", None),
        "status": str(getattr(info, "status", "unknown")),
    }
