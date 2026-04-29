#!/usr/bin/env bash
# Stub — wires up in P3.
# Will: ingest each open dataset (Penn Action, THETIS, GolfDB, Fitness-AQA/FLEX),
# run pipeline (P1 + P2) over each clip, write manifest.jsonl, then upsert to Qdrant (P4).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "build_library.sh: not yet implemented (lands in P3)"
exit 0
