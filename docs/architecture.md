# MimicCoach architecture

## Data flow

```
phone MP4
    │
    ▼
MediaPipe Pose Landmarker  ──►  per-frame 33-keypoint sequence
    │
    ▼
skeleton_map (33 → H36M-17)
    │
    ▼
MotionBERT          ──►  per-frame 512-d embeddings
    │
    ▼
phase segmenter     ──►  (phase_name, start_frame, end_frame) tuples
    │                    [rule-based velocity zero-crossings]
    ▼
mean-pool over each phase ─► 5–6 phase tokens × 512-d, L2-normalized
    │
    ▼
Qdrant query (multivector + payload filter)
    │   collection = "motions"
    │   comparator = MAX_SIM
    │   filter = {sport, motion, skill_level?, body_type?}
    ▼
top-k pro matches  ─►  per-phase MaxSim breakdown  ─►  rule-based coaching tip
                                                              │
                                                              ▼
                                              JSON to frontend → side-by-side
                                              <video> + <canvas> skeleton overlay
```

## Module map

### Backend (Python, runs on Modal)

```
backend/
  app.py                          # Modal app + FastAPI endpoints
  pipeline/
    pose_extract.py               # MediaPipe wrapper → per-frame landmarks
    skeleton_map.py               # MediaPipe-33 → H36M-17 mapping
    embed.py                      # MotionBERT loader + per-phase mean-pool
    segment.py                    # Velocity zero-crossing phase segmenter
    motions.yaml                  # Phase definitions (source of truth)
    body_type.py                  # Shoulder/hip ratio bucketing
    coach.py                      # Rule-based coaching tips
  qdrant_io/
    client.py                     # Env-driven Qdrant client
    schema.py                     # create_collection (multivector + payload index)
    upsert.py                     # Bulk upsert manifest.jsonl
    query.py                      # Multivector + payload-filter query
  reference/
    ingest_penn.py                # Penn Action loader
    ingest_thetis.py              # THETIS loader
    ingest_golfdb.py              # GolfDB loader
    ingest_fitness.py             # Fitness-AQA / FLEX loader
    ingest_youtube.py             # yt-dlp + CC filter + pose-only retention
  tests/                          # pytest suite
```

### Frontend (Next.js 15, runs on Vercel)

```
frontend/
  app/
    page.tsx                      # Upload page
    analyze/[id]/page.tsx         # Result page
    api/proxy/route.ts            # Passthrough to Modal
    layout.tsx
    globals.css
  components/
    SplitVideo.tsx                # Two synced <video> elements
    SkeletonCanvas.tsx            # 33-keypoint skeleton overlay
    PhaseScores.tsx               # Per-phase score chips
    CoachingTip.tsx               # One-line tip card
    FilterBar.tsx                 # sport / skill_level / body_type
  lib/
    api.ts                        # fetch wrapper to /api/proxy
    poseDraw.ts                   # canvas drawing helpers
```

### Ops

```
scripts/
  build_library.sh                # one-shot reference library build
  warm_modal.sh                   # cron-callable warm-keep
.github/workflows/
  ci.yml                          # ruff + pytest + pnpm typecheck + pnpm build
```

## Qdrant schema

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, MultiVectorConfig,
    MultiVectorComparator, HnswConfigDiff,
)

client.create_collection(
    collection_name="motions",
    vectors_config={
        "phase_tokens": VectorParams(
            size=512,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=HnswConfigDiff(m=0),
        ),
    },
)

# Payload field indexes (for fast filtering)
for field, schema in [
    ("sport", "keyword"),
    ("motion", "keyword"),
    ("skill_level", "keyword"),
    ("body_type", "keyword"),
]:
    client.create_payload_index("motions", field_name=field, field_schema=schema)
```

A point's `phase_tokens` is a list of 5–6 vectors (one per phase). Storage budget: 7 motions × 50 clips × 6 tokens × 512 dim × 4 B ≈ 4.3 MB raw vectors + 18 MB payloads → comfortably under the 1 GB free tier.
