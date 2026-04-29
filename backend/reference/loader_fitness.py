"""Stub loader for Fitness-AQA / FLEX.

Both projects publish weight-room exercise videos with skill-level tier
labels. Fitness-AQA (Parmar et al.) covers BackSquat, OverheadPress,
BarbellRow; FLEX (2025) extends to 20 weight-loaded exercises with
multi-view RGB + 3D pose + sEMG and a 3-tier skill labeling.

References:
  * Fitness-AQA — https://github.com/ParitoshParmar/Fitness-AQA
  * FLEX        — https://haoyin116.github.io/FLEX_Dataset/

To wire this in:
  1. Pick whichever has accessible direct downloads (likely FLEX via
     Zenodo or Fitness-AQA's release link).
  2. For each motion (squat, bench, row), filter to the relevant
     exercise tag.
  3. Map the dataset's skill tier → our {beginner, intermediate, pro}.
  4. Run MediaPipe → embed → emit manifest row.

Until then this loader yields nothing.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_rows(_data_root: Path | None = None) -> Iterator[dict[str, Any]]:
    """Yield manifest rows from a local Fitness-AQA/FLEX checkout. NOT YET IMPLEMENTED."""
    return iter(())
