"""Push a manifest.jsonl into the configured Qdrant collection.

CLI:

    python -m reference.seed_qdrant [--manifest path] [--recreate]

Reads QDRANT_URL / QDRANT_API_KEY / QDRANT_PATH from env (see qdrant_io
.client.make_client). Creates the collection if missing, then bulk-
upserts the manifest.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from qdrant_io.client import make_client
from qdrant_io.schema import COLLECTION_NAME, create_collection
from qdrant_io.upsert import upsert_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Qdrant with the reference manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).parent / "manifest.jsonl",
        help="Path to the manifest.jsonl produced by reference.bootstrap.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop the existing collection before upserting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(
            f"manifest not found at {args.manifest}. "
            f"Run `python -m reference.bootstrap` first."
        )

    client = make_client()
    create_collection(client, recreate=args.recreate)
    n = upsert_manifest(client, args.manifest, batch_size=args.batch_size)
    print(f"upserted {n} points into {COLLECTION_NAME!r}")


if __name__ == "__main__":
    main()
