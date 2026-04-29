# MimicCoach

> Self-coaching by pose-embedding lookup. Upload a phone video of yourself doing a tennis serve, squat, or golf swing — see how the closest pro does it, side-by-side, with per-phase scores and a coaching tip on your weakest phase.

Submission for the Qdrant **Think Outside the Bot** virtual hackathon.

## What makes this different

Most pose-search demos average a whole motion into one vector and lose the *where*. MimicCoach stores each motion as **5–6 phase tokens per clip** in a Qdrant multivector point and queries it with **late-interaction MaxSim** — the same trick ColBERT and ColPali use for documents, applied to motion. The result: retrieval is sensitive to *which* phase of your serve is off, not just whether the whole serve looks vaguely like a pro's.

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│ Phone video  │ →  │  MediaPipe   │ →  │  MotionBERT  │ →  │  Phase tokens    │
│  (MP4)       │    │  Pose (33)   │    │ (512-d/frame)│    │ (5–6 × 512-d)    │
└──────────────┘    └──────────────┘    └──────────────┘    └────────┬─────────┘
                                                                      │
                          ┌──────────────────────────────────────────┘
                          ▼
                ┌──────────────────────┐         ┌─────────────────────┐
                │  Qdrant multivector  │   ←─→   │  Pro reference lib  │
                │  collection          │         │  (Penn / THETIS /   │
                │  MaxSim comparator   │         │   GolfDB / FLEX)    │
                │  + payload filters   │         └─────────────────────┘
                └──────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │  Top-k pro match +     │
              │  per-phase scores +    │
              │  coaching tip          │
              └────────────────────────┘
```

## Qdrant features showcased

- **Late-interaction multivector storage** with `MultiVectorConfig(comparator=MAX_SIM)` — one point per pro clip carries a list of per-phase token vectors.
- **MaxSim scoring** at query time produces per-phase similarity, not just an aggregate.
- **Payload-indexed filtering** on `sport`, `motion`, `skill_level`, and `body_type` runs unchanged alongside the multivector query.

## Supported motions

| Sport | Motions |
|---|---|
| Tennis | Serve (hero) · Forehand · Backhand |
| Fitness | Barbell back squat · Bench press · Bent-over row |
| Golf | Full swing |

## Run locally

Requires Python 3.11+, Node 20+, [uv](https://github.com/astral-sh/uv), and [pnpm](https://pnpm.io).

```bash
# 1. Install backend deps (creates .venv automatically)
cd backend && uv sync

# 2. Install frontend deps
cd ../frontend && pnpm install

# 3. Set up environment (see backend/.env.example, frontend/.env.local.example)

# 4. Build the reference library (one-shot)
./scripts/build_library.sh

# 5. Run dev servers
modal serve backend/app.py    # in one terminal
pnpm --dir frontend dev       # in another
```

Open [http://localhost:3000](http://localhost:3000) and upload a clip.

## Deployment

- Backend: [Modal](https://modal.com) (`modal deploy backend/app.py`)
- Frontend: [Vercel](https://vercel.com) (auto-deploys from `frontend/`)
- Vector DB: [Qdrant Cloud](https://cloud.qdrant.io) free tier

## Project layout

See [`docs/architecture.md`](docs/architecture.md) for the full module map.

## Credits & dataset attributions

- **Penn Action** — University of Pennsylvania
- **THETIS** — Three-Dimensional Tennis Shot Recognition
- **GolfDB** — McNally et al., 2019
- **Fitness-AQA / FLEX** — Parmar et al.
- **MotionBERT** — Walter Zhu et al., ICCV 2023 (Apache-2.0)
- **MediaPipe Pose Landmarker** — Google (Apache-2.0)
- **Qdrant** — qdrant.tech (Apache-2.0)

YouTube-sourced clips (where used) are limited to Creative Commons content, with `source_url` and `license_note` recorded in the Qdrant payload. Source MP4s are not redistributed; only pose JSON is retained.

## License

MIT — see [LICENSE](LICENSE).
