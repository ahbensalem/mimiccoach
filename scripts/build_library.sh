#!/usr/bin/env bash
# Build the MimicCoach reference library and load it into Qdrant.
#
# Default behavior (no args): generates the synthetic library — 216 clips
# spanning all 7 motions with varied skill_level and body_type buckets,
# zero external downloads required. The synthetic library is enough to
# demo every Qdrant-side feature; real data is a quality upgrade layered
# on top once it's been acquired (see docs/data-acquisition.md).
#
# Real-data sources are wired by editing backend/reference/bootstrap.py
# to include the matching loader_* iterators.
#
# Environment:
#   QDRANT_URL      — when set, seeds Qdrant Cloud directly.
#   QDRANT_API_KEY  — paired with QDRANT_URL.
#   QDRANT_PATH     — local on-disk Qdrant (e.g. ./.qdrant-data).
#                     If neither QDRANT_URL nor QDRANT_PATH is set, the
#                     seed step uses the in-memory client and the
#                     manifest stays on disk for future loading.
set -euo pipefail
cd "$(dirname "$0")/.."

cd backend

# When the GolfDB downloads have placed a local MediaPipe model at the
# canonical location, point pose_extract at it so iter_rows() works
# outside the Modal image (which bakes the same .task file at /opt).
if [[ -z "${MEDIAPIPE_MODEL_PATH:-}" ]]; then
  candidate="reference/data/golfdb/pose_landmarker_full.task"
  if [[ -f "$candidate" ]]; then
    export MEDIAPIPE_MODEL_PATH="$PWD/$candidate"
    echo "==> Using local MediaPipe model at $MEDIAPIPE_MODEL_PATH"
  fi
fi

echo "==> Generating manifest (source: ${MIMICCOACH_SOURCE:-all})…"
uv run python -m reference.bootstrap --source "${MIMICCOACH_SOURCE:-all}"

if [[ -n "${QDRANT_URL:-}" || -n "${QDRANT_PATH:-}" ]]; then
  echo "==> Seeding Qdrant…"
  uv run python -m reference.seed_qdrant --recreate
else
  echo "==> Skipping Qdrant seed (set QDRANT_URL or QDRANT_PATH to enable)."
  echo "    Manifest is on disk at backend/reference/manifest.jsonl"
fi

echo "==> Done."
