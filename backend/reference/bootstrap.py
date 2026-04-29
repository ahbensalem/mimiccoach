"""Build the reference manifest.jsonl from configured sources.

Sources can be mixed and matched. The synthetic source is always
available and is the default — it requires no external downloads,
registrations, or licensing review, and gives the demo a baseline
~210 reference rows across all 7 motions immediately.

Real-data loaders (Penn Action, THETIS, GolfDB, Fitness-AQA, YouTube
CC) live as siblings in this package and can be wired in once the
underlying data is on disk. See docs/data-acquisition.md for the
manual download path.

CLI:

    python -m reference.bootstrap [--out path] [--source synthetic|all]

Default: writes ./reference/manifest.jsonl with the synthetic library.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .synthetic import SyntheticEntry
from .synthetic import generate as generate_synthetic


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


def iter_rows(source: str) -> Iterator[dict[str, Any]]:
    if source in ("synthetic", "all"):
        for e in generate_synthetic():
            yield _to_manifest_row(e)
    # Future: penn / thetis / golfdb / fitness loaders go here once real
    # data is on disk. They emit the same row shape as the synthetic source.


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
        default="synthetic",
        choices=("synthetic", "all"),
        help="Which loader(s) to run.",
    )
    args = parser.parse_args()

    n = write_manifest(iter_rows(args.source), args.out)
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
