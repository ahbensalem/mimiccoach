"""Stub loader for the Penn Action dataset.

Penn Action covers tennis_serve, golf_swing, bench_press, squat (and
many other actions). Per-frame 13-keypoint pose annotations + raw
RGB videos. Project page: http://dreamdragon.github.io/PennAction/

To wire this in:
  1. Download Penn-Action-Dataset.zip and extract under
     backend/reference/data/penn-action.
  2. For each clip in the train_test_split.txt, load the video,
     run MediaPipe Pose to get our 33-landmark sequence (Penn's
     13-keypoint annotations are too sparse for our pipeline),
     run segmentation, embed phases, and emit a manifest row.
  3. skill_level: clips of named pros (visible in metadata) → "pro",
     others default to "intermediate".

Until then this loader yields nothing.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_rows(_data_root: Path | None = None) -> Iterator[dict[str, Any]]:
    """Yield manifest rows from a local Penn Action checkout. NOT YET IMPLEMENTED."""
    return iter(())
