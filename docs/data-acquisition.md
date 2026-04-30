# Reference library — data acquisition

MimicCoach ships with a **synthetic reference library** as the default
data source. It needs no external downloads, no registrations, and no
license review: every clip is procedurally generated from the same
pose-extraction + segmentation + embedding pipeline that user uploads
go through, so retrieval against synthetic references behaves exactly
the way it would against real clips.

The synthetic library covers all 7 motions with:

- 6 body templates × 5–6 variations per motion = ~210 clips
- A balanced skill_level distribution (beginner / intermediate / pro)
- All three body_type buckets represented per motion

That's enough for the demo to show every Qdrant-side feature
(multivector + MaxSim, payload-indexed filters, per-phase scores).

## Build the synthetic library

```bash
./scripts/build_library.sh
```

This writes `backend/reference/manifest.jsonl` and (when
`QDRANT_URL` or `QDRANT_PATH` is set) seeds the `motions` collection.

## Real-data upgrade path

Real reference clips are a quality upgrade layered on top of the
synthetic baseline. The same `manifest.jsonl` shape flows through
Qdrant identically once a loader is wired into
`backend/reference/bootstrap.py:iter_rows()`. When a real loader
emits rows for a motion, the synthetic generator skips that motion
automatically — so on a current-state machine, golf retrieval is
**100% real** as soon as `scripts/download_golfdb.sh` has run.

| Source | Motions covered | Data type | Loader | Status |
|---|---|---|---|---|
| **GolfDB** | golf full swing | 1,400 clips, 8 ground-truth event labels (CC BY-NC 4.0) | `loader_golfdb.py` | **Wired** — auto-detected when `backend/reference/data/golfdb/` has metadata + at least one video |
| **Penn Action** | tennis_serve, golf_swing, bench_press, squat | RGB videos + 13-keypoint annotations | `loader_penn.py` | Stub |
| **THETIS** | tennis serve / forehand / backhand | RGB + depth + 2D/3D skeleton | `loader_thetis.py` | Stub |
| **Fitness-AQA / FLEX** | barbell squat / bench press / bent-over row | RGB + skill-tier labels | `loader_fitness.py` | Stub |
| **YouTube CC** | gap-filling (esp. bent-over row) | yt-dlp + Creative Commons filter | `loader_youtube.py` | Stub |

### GolfDB

```bash
./scripts/download_golfdb.sh   # pulls metadata + MediaPipe model + a curated
                               # subset of YouTube source videos (~10 videos,
                               # yields ~50–80 swings after cropping)
./scripts/build_library.sh     # rebuild manifest — synthetic golf is now
                               # excluded automatically; only real GolfDB
                               # rows are emitted for golf_swing
```

The 18 GB pre-cropped `videos_160.zip` artifact from the GolfDB README
is also supported — drop it under
`backend/reference/data/golfdb/videos_160/` and the loader will skip the
in-loader cropping step. Either layout produces the same manifest rows.

Each loader has a `iter_rows()` function that yields
`{id, phase_tokens, payload}` dicts in the same shape the synthetic
generator produces. The bootstrap orchestrator calls them in order;
manifest IDs should be globally unique across sources.

## License hygiene

Every manifest row carries `license_note` and (where applicable)
`source_url` in the payload, both indexed by Qdrant. The README
attributes every dataset by name. For YouTube-sourced material the
default policy is **pose-only retention**: extract pose JSON, then
delete the source MP4 — nothing copyrightable is redistributed.
