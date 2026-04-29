# MimicCoach

> **Self-coaching by pose-embedding lookup.** Upload a phone clip of yourself doing a tennis serve, squat, or golf swing — see how the closest pro does it, side-by-side with phase-by-phase scores and a coaching tip on the part of your motion that needs the most work.

Submission for the **Qdrant *Think Outside the Bot*** virtual hackathon.

---

## What this builds with Qdrant

> Most pose-search demos average a whole motion into one vector and lose the *where*. We don't.

MimicCoach stores each motion as **5–6 phase tokens** in a Qdrant **multivector point** and queries it with **late-interaction MaxSim** — the same trick ColBERT and ColPali use for documents, applied to motion. Retrieval becomes sensitive to *which* phase of your serve is off, not just whether the whole serve looks vaguely like a pro's.

Three Qdrant features carry the weight:

| Feature | How we use it |
|---|---|
| **`MultiVectorConfig(comparator=MAX_SIM)`** | Each pro clip is one Qdrant point that holds a *list* of 512-d phase tokens. The server computes per-token max-similarity and aggregates — exactly the late-interaction primitive ColBERT introduced for text. |
| **Per-phase score breakdown** | We re-derive MaxSim per query token from the matched stored tokens client-side, surfacing per-phase score chips directly to the user. The weakest phase drives the rule-based coaching tip. |
| **Payload-indexed filters** | `sport`, `motion`, `skill_level`, `body_type` are keyword-indexed and combined into a `Filter` clause that runs alongside the multivector query. Toggling a chip in the UI literally swings the top match — that's the live filter demo beat. |

Code: [`backend/qdrant_io/schema.py`](backend/qdrant_io/schema.py) · [`backend/qdrant_io/query.py`](backend/qdrant_io/query.py).

---

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Phone video  │ →  │  MediaPipe   │ →  │  Pose-and-motion │ →  │  Phase tokens    │
│  (MP4)       │    │  Pose (33)   │    │   embedder (512) │    │ (5–6 × 512-d)    │
└──────────────┘    └──────────────┘    └──────────────────┘    └────────┬─────────┘
                                                                          │
                       ┌──────────────────────────────────────────────────┘
                       ▼
          ┌──────────────────────────┐         ┌─────────────────────────────┐
          │  Qdrant 'motions'        │   ←─→   │  Reference library          │
          │  multivector collection  │         │  - synthetic baseline (216) │
          │  comparator=MAX_SIM      │         │  - + Penn / THETIS / GolfDB │
          │  + payload filters       │         │    / FLEX / YouTube CC      │
          └──────────┬───────────────┘         └─────────────────────────────┘
                     ▼
       ┌──────────────────────────────┐
       │  Top-k pro match +           │
       │  per-phase MaxSim scores +   │
       │  rule-based coaching tip     │
       └──────────────────────────────┘
```

Detailed module layout: [`docs/architecture.md`](docs/architecture.md) · phase definitions: [`docs/motions.md`](docs/motions.md) · data path: [`docs/data-acquisition.md`](docs/data-acquisition.md).

## Supported motions

| Sport | Motions |
|---|---|
| **Tennis** | Serve ★ · Forehand · Backhand |
| **Fitness** | Barbell back squat · Bench press · Bent-over row |
| **Golf** | Full swing |

★ = hero motion (anchors the demo video, gets the most polish on per-phase tuning).

Phase boundaries per motion are encoded in [`backend/pipeline/motions.yaml`](backend/pipeline/motions.yaml) — the schema-of-truth read directly by the segmenter.

---

## Run locally (no external accounts required)

Requirements: Python 3.11+, Node 20+, [`uv`](https://github.com/astral-sh/uv), [`pnpm`](https://pnpm.io).

```bash
# 1. Install
cd backend && uv sync && cd ../frontend && pnpm install && cd ..

# 2. Build the synthetic reference library
#    (216 clips spanning all 7 motions; no downloads, no registrations).
QDRANT_PATH=$PWD/.qdrant-data ./scripts/build_library.sh

# 3. Start the backend in dev mode (watches for changes)
modal serve backend/app.py
# → prints something like https://<workspace>--mimiccoach-fastapi-app-dev.modal.run

# 4. Configure the frontend to point at the printed URL
echo 'MODAL_BACKEND_URL=<paste-the-url>' > frontend/.env.local

# 5. Start the frontend
pnpm --dir frontend dev
```

Open [http://localhost:3000](http://localhost:3000), upload a clip, see the result.

To use real reference data (Penn Action / THETIS / GolfDB / Fitness-AQA / YouTube CC), see [`docs/data-acquisition.md`](docs/data-acquisition.md).

## Deploy to production

See [`DEPLOY.md`](DEPLOY.md) for the Modal + Vercel + Qdrant Cloud runbook.

Scheduled `warm_keep` keeps the Modal container hot during demo windows (every 4 minutes), so live judging doesn't see a 30-second cold-start.

---

## Demo flow (3-minute video storyboard)

1. **Cold open (0:00–0:15)** — split-screen, the user's amateur tennis serve next to the closest pro, skeletons overlaid, color-coded by phase.
2. **Per-phase scores (0:15–0:45)** — score chips fade in: `Toss 0.78 · Trophy 0.62 ← weakest · Contact 0.84 · Follow-through 0.91`. The coaching-tip card pops with the rule-based one-liner ("Trophy position needs more depth — coil the back, get the racket head higher behind you.").
3. **All 7 motions (0:45–1:30)** — quick montage: squat, bench, row, golf, forehand, backhand, each scoring the user against a pro.
4. **Architecture beat (1:30–2:15)** — animated explainer of multivector storage and MaxSim querying. Soundbite: *"This is what ColBERT does for text — we are doing it for motion."*
5. **Filter demo + close (2:15–3:00)** — toggle skill_level / body_type chips on the UI; the top match changes live. Closing card with the GitHub URL.

---

## Project layout

```
backend/                Python — runs on Modal
├── app.py                FastAPI ASGI app + analyze_from_landmarks()
├── pipeline/             pose extraction, segmentation, embedding
│   ├── pose_extract.py
│   ├── skeleton_map.py   MediaPipe-33 ↔ H36M-17
│   ├── embed.py          per-frame 512-d + per-phase mean-pool
│   ├── segment.py        rule-based phase segmenter
│   ├── motions.yaml      phase definitions
│   ├── body_type.py      shoulder/hip ratio bucketing
│   └── coach.py          rule-based coaching tips
├── qdrant_io/            schema, upsert, multivector + filter query
├── reference/            synthetic generator + real-data loader stubs
└── tests/                pytest — 76 tests, ruff clean

frontend/               Next.js 15 + Tailwind — runs on Vercel
├── app/                  upload page, analyze result, /api/proxy/*
├── components/           SplitVideo, SkeletonCanvas, PhaseScores, CoachingTip, FilterBar
└── lib/                  API client, types, canvas helpers

scripts/                build_library.sh, warm_modal.sh
docs/                   architecture · motions · data-acquisition
.github/workflows/      ci.yml (ruff + pytest + pnpm typecheck + pnpm build)
```

## Test coverage

```
backend:  76 pytest tests, ruff clean
  - skeleton_map (7) · embed (9) · segment (13) · qdrant (10)
  - body_type (5) · coach (5) · synthetic_library (8)
  - smoke (3) · analyze (4) · all_motions_e2e (7) · filter_swing (3)

frontend: tsc --noEmit clean, eslint clean, next build passes
```

---

## Credits & dataset attributions

- **Qdrant** — qdrant.tech (Apache-2.0)
- **MediaPipe Pose Landmarker** — Google (Apache-2.0)
- **MotionBERT** — Walter Zhu et al., ICCV 2023 (Apache-2.0; planned swap-in, see [`backend/pipeline/embed.py`](backend/pipeline/embed.py))
- **Penn Action** — University of Pennsylvania
- **THETIS** — Three-Dimensional Tennis Shot Recognition (CVPR-W 2013)
- **GolfDB** — McNally et al., 2019
- **Fitness-AQA / FLEX** — Parmar et al.

YouTube-sourced clips (where used) are limited to Creative Commons content. Default policy is **pose-only retention** — `source_url` and `license_note` recorded in the Qdrant payload, the source MP4 deleted after pose extraction.

## License

MIT — see [LICENSE](LICENSE).
