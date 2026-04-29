"""Stub loader for GolfDB.

GolfDB ships ~1,400 golf swing clips with 8 ground-truth event labels
per swing (Address, Toe-Up, Mid-Backswing, Top, Mid-Downswing, Impact,
Mid-Follow-Through, Finish). Public: https://github.com/wmcnally/golfdb

To wire this in:
  1. git clone https://github.com/wmcnally/golfdb backend/reference/data/golfdb
  2. Download the cropped 160×160 video shards from the README's release
     link and extract under data/golfdb/videos.
  3. For each clip, use the ground-truth events to derive 6 phase
     boundaries (collapse 8 events to our 6 phases:
        address      → Address
        backswing    → Address → Mid-Backswing
        top          → Mid-Backswing → Top
        downswing    → Top → Mid-Downswing
        impact       → Mid-Downswing → Impact
        finish       → Impact → Finish
     ).
  4. Run MediaPipe → embed → emit manifest row.

Until then this loader yields nothing.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_rows(_data_root: Path | None = None) -> Iterator[dict[str, Any]]:
    """Yield manifest rows from a local GolfDB checkout. NOT YET IMPLEMENTED."""
    return iter(())
