"""Qdrant client factory.

Resolves connection settings from the environment so the same code path
serves local dev (`:memory:`), embedded on-disk testing, and Qdrant Cloud
production.

Precedence:
  1. QDRANT_URL set        → remote (Cloud or self-hosted)
  2. QDRANT_PATH set       → embedded on-disk Qdrant
  3. otherwise             → in-memory (good for unit tests)
"""
from __future__ import annotations

import os

from qdrant_client import QdrantClient


def make_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL")
    if url:
        return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))

    path = os.environ.get("QDRANT_PATH")
    if path:
        return QdrantClient(path=path)

    return QdrantClient(":memory:")
