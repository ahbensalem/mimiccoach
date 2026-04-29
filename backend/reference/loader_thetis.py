"""Stub loader for the THETIS tennis dataset.

THETIS provides depth + RGB + 2D/3D skeleton recordings for tennis
serve, forehand, backhand (plus other strokes we don't use). Public
GitHub: https://github.com/THETIS-dataset/dataset

To wire this in:
  1. git clone https://github.com/THETIS-dataset/dataset backend/reference/data/thetis
  2. Inspect the skeleton file format (the 2D pose annotations are what
     we need; we map them onto MediaPipe-33 indices via the closest joint).
  3. For each clip, derive the phase boundaries (THETIS is unsegmented,
     so run our segmenter or use the provided 'shot' midpoint annotations).
  4. Emit one manifest row per clip with skill_level taken from the
     subject code (THETIS encodes expert vs beginner in subject IDs).

Until then this loader yields nothing — bootstrap.py uses the synthetic
source as the default.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_rows(_data_root: Path | None = None) -> Iterator[dict[str, Any]]:
    """Yield manifest rows from a local THETIS checkout. NOT YET IMPLEMENTED."""
    return iter(())
