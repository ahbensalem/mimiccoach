"""Bulk upsert of pose-token reference points into the `motions` collection.

Each manifest entry is one clip; its `phase_tokens` is a list of 512-d
unit vectors (one per phase). Payload carries sport / motion /
skill_level / body_type / athlete / source_url / license_note. The
upsert uses Qdrant's batch API in groups of `batch_size` to keep
gRPC payloads small.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from .schema import COLLECTION_NAME, PHASE_TOKENS_FIELD


def iter_manifest(manifest_path: Path) -> Iterator[dict[str, Any]]:
    """Yield one parsed JSON object per non-empty line of a JSONL manifest."""
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def manifest_to_points(entries: list[dict[str, Any]]) -> list[PointStruct]:
    """Map manifest entries to Qdrant PointStructs.

    Each entry must have:
      * id            : int
      * phase_tokens  : list[list[float]] (one inner list per phase)
      * payload       : dict
    """
    points: list[PointStruct] = []
    for e in entries:
        if "id" not in e or "phase_tokens" not in e or "payload" not in e:
            raise ValueError(f"manifest entry missing required keys: {sorted(e)}")
        points.append(
            PointStruct(
                id=e["id"],
                vector={PHASE_TOKENS_FIELD: e["phase_tokens"]},
                payload=e["payload"],
            )
        )
    return points


def upsert_points(
    client: QdrantClient,
    points: list[PointStruct],
    *,
    batch_size: int = 64,
) -> int:
    """Upsert points in batches. Returns the number upserted."""
    total = 0
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
        total += len(batch)
    return total


def upsert_manifest(
    client: QdrantClient,
    manifest_path: Path,
    *,
    batch_size: int = 64,
) -> int:
    """Stream a manifest.jsonl file into the collection. Returns the count."""
    entries = list(iter_manifest(manifest_path))
    points = manifest_to_points(entries)
    return upsert_points(client, points, batch_size=batch_size)
