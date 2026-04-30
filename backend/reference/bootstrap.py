"""Build the reference manifest.jsonl from configured sources.

The synthetic generator covers all 7 motions out of the box. As real-data
loaders come online they take over their motion(s) and the synthetic
source is excluded for those motions automatically — so once GolfDB is
wired in, every golf row in the manifest is real.

Real-data loaders (Penn Action, THETIS, GolfDB, Fitness-AQA, YouTube
CC) live as siblings in this package and can be wired in once the
underlying data is on disk. See docs/data-acquisition.md for the
manual download path.

CLI:

    python -m reference.bootstrap [--out path] [--source synthetic|all|golfdb-only]

Default: writes ./reference/manifest.jsonl. Real loaders are auto-
enabled when their ``data/<source>`` directory contains usable inputs.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from . import loader_golfdb
from .synthetic import SyntheticEntry
from .synthetic import generate as generate_synthetic

logger = logging.getLogger(__name__)

# Motions that the synthetic generator emits. Real-data sources subtract
# from this set.
_SYNTHETIC_MOTIONS: frozenset[str] = frozenset(
    {
        "tennis_serve", "tennis_forehand", "tennis_backhand",
        "fitness_squat", "fitness_bench_press", "fitness_bent_over_row",
        "golf_swing",
    }
)


def _to_manifest_row(entry: SyntheticEntry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "phase_tokens": entry.phase_tokens,
        "payload": entry.payload,
    }


def write_manifest(rows: Iterable[dict[str, Any]], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _golfdb_available() -> bool:
    """Cheap probe: do we have GolfDB metadata + at least one video on disk?"""
    root = loader_golfdb._resolve_data_root(None)
    if not (root / "golfDB.mat").exists() and not (root / "golfDB.pkl").exists():
        return False
    has_cropped = (root / "videos_160").is_dir() and any(
        (root / "videos_160").glob("*.mp4")
    )
    has_source = (root / "videos").is_dir() and any(
        p for p in (root / "videos").iterdir() if p.is_file()
    )
    return has_cropped or has_source


def iter_rows(source: str) -> Iterator[dict[str, Any]]:
    """Yield manifest rows for the requested source(s).

    Source modes:
      * ``synthetic`` — synthetic only, every motion (legacy behavior).
      * ``all`` — real loaders take their motion(s) when their data is
        on disk; synthetic fills the rest. This is the demo-time mode.
      * ``golfdb-only`` — GolfDB only (debug / single-source seeding).
    """
    real_motions: set[str] = set()

    if source in ("all", "golfdb-only") and _golfdb_available():
        yield from loader_golfdb.iter_rows()
        real_motions.add("golf_swing")
    elif source == "all" and not _golfdb_available():
        logger.info(
            "golfdb data not on disk; falling back to synthetic golf "
            "(set MIMICCOACH_GOLFDB_ROOT and run scripts/download_golfdb.sh)"
        )

    if source == "golfdb-only":
        return

    if source in ("synthetic", "all"):
        for e in generate_synthetic():
            if e.payload.get("motion") in real_motions:
                continue  # a real loader covers this motion
            yield _to_manifest_row(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the MimicCoach reference manifest.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "manifest.jsonl",
        help="Where to write the manifest (default: backend/reference/manifest.jsonl)",
    )
    parser.add_argument(
        "--source",
        default="all",
        choices=("synthetic", "all", "golfdb-only"),
        help="Which loader(s) to run.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    n = write_manifest(iter_rows(args.source), args.out)
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
