"""Stub loader for Creative-Commons YouTube clips.

For motions where the open datasets thin out (notably bent-over row),
yt-dlp can fill the gap — restricted to Creative Commons-licensed
content via `--match-filter "license=Creative Commons"`. We default
to a pose-only retention policy: extract pose JSON, then delete the
source MP4 to sidestep redistribution risk.

To wire this in:
  1. Maintain a hand-curated list of CC YouTube URLs in
     `backend/reference/youtube_curated.txt`.
  2. Run yt-dlp to download each into a tmp dir.
  3. For each, run MediaPipe → embed → emit manifest row with
     {source: "youtube_cc", source_url: <url>, license_note: "CC-BY"
     (or whichever variant)}.
  4. Delete the MP4 immediately after pose extraction.

Until then this loader yields nothing.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_rows(_curated_list: Path | None = None) -> Iterator[dict[str, Any]]:
    """Yield manifest rows from a curated CC YouTube list. NOT YET IMPLEMENTED."""
    return iter(())
