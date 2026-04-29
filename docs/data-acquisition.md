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
synthetic baseline. The loader stubs are in
`backend/reference/loader_*.py`; each has the manual-acquisition
steps documented in its docstring. The same `manifest.jsonl` shape
flows through Qdrant identically once you wire the loader into
`backend/reference/bootstrap.py:iter_rows()`.

| Source | Motions covered | Data type | Loader |
|---|---|---|---|
| **Penn Action** | tennis_serve, golf_swing, bench_press, squat | RGB videos + 13-keypoint annotations | `loader_penn.py` |
| **THETIS** | tennis serve / forehand / backhand | RGB + depth + 2D/3D skeleton | `loader_thetis.py` |
| **GolfDB** | golf full swing | 1,400 clips, 8 ground-truth event labels | `loader_golfdb.py` |
| **Fitness-AQA / FLEX** | barbell squat / bench press / bent-over row | RGB + skill-tier labels | `loader_fitness.py` |
| **YouTube CC** | gap-filling (esp. bent-over row) | yt-dlp + Creative Commons filter | `loader_youtube.py` |

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
