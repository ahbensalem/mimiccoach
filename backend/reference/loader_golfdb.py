"""GolfDB loader.

GolfDB ships ~1,400 golf swing clips with 8 ground-truth event labels per
swing (Address, Toe-Up, Mid-Backswing, Top, Mid-Downswing, Impact,
Mid-Follow-Through, Finish). Public:
https://github.com/wmcnally/golfdb (CC BY-NC 4.0).

This loader emits one manifest row per processable annotation. Pose
extraction is real (MediaPipe Tasks API) — there is no synthetic shortcut
on the golf path, so once a manifest contains golfdb rows, golf
retrieval is 100% real-data.

Data layout (all under ``data_root``, gitignored):

    golfDB.mat                 # metadata, downloaded from the GolfDB repo
    videos_160/{anno_id}.mp4   # cropped 160×160 clips (preferred — official
                                 GolfDB release artifact, README points to a
                                 Google Drive zip with all 1,400 of them)
    videos/{youtube_id}.mp4    # raw YouTube source video (fallback layout —
                                 we crop+slice in-loader using bbox+events
                                 from the metadata; useful when only a
                                 subset has been pulled with yt-dlp)

Phase mapping — GolfDB's 10-element events array is
``[start_pad, address, toe_up, mid_back, top, mid_down, impact,
mid_follow, finish, end_pad]`` (indices 0 and 9 bracket the clip with
padding). We collapse to MimicCoach's 6 golf phases:

    address    : start_pad → address
    backswing  : address   → mid_backswing
    top        : mid_back  → top
    downswing  : top       → mid_downswing
    impact     : mid_down  → impact
    finish     : impact    → end_pad

Pose landmarks are cached as ``pose_cache/{entry_id}.npz`` alongside the
manifest row so the live overlay can reload them via
``landmarks_for_entry()`` without re-running MediaPipe at request time.
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.body_type import body_type_bucket
from pipeline.embed import HandCraftedEmbedder, phase_token
from pipeline.pose_extract import PoseExtractor

logger = logging.getLogger(__name__)


# Globally-unique manifest IDs need to avoid colliding with synthetic
# (1..~250). GolfDB annotation ids are 0..1399, so 100_000+id keeps us
# clear of every other source we plan to wire.
ID_OFFSET: int = 100_000

PHASE_NAMES: tuple[str, ...] = (
    "address", "backswing", "top", "downswing", "impact", "finish",
)

# Index pairs into the 10-element GolfDB events vector that delimit each
# of the 6 MimicCoach phases (half-open intervals).
_PHASE_BOUNDS: tuple[tuple[int, int], ...] = (
    (0, 1),  # address    : pad-start → Address
    (1, 3),  # backswing  : Address    → Mid-Backswing
    (3, 4),  # top        : Mid-Back   → Top
    (4, 5),  # downswing  : Top        → Mid-Down
    (5, 6),  # impact     : Mid-Down   → Impact
    (6, 9),  # finish     : Impact     → pad-end
)


@dataclass(frozen=True)
class GolfDBAnno:
    anno_id: int
    youtube_id: str
    player: str
    sex: str
    club: str
    view: str
    slow: int
    events: tuple[int, ...]      # 10 absolute frame indices in source video
    bbox: tuple[float, float, float, float]
    split: int


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def _load_metadata_mat(path: Path) -> list[GolfDBAnno]:
    from scipy.io import loadmat

    mat = loadmat(str(path))
    arr = mat["golfDB"][0]
    out: list[GolfDBAnno] = []
    for r in arr:
        out.append(
            GolfDBAnno(
                anno_id=int(r["id"][0, 0]),
                youtube_id=str(r["youtube_id"][0]),
                player=str(r["player"][0]),
                sex=str(r["sex"][0]),
                club=str(r["club"][0]),
                view=str(r["view"][0]),
                slow=int(r["slow"][0, 0]),
                events=tuple(int(x) for x in r["events"][0]),
                bbox=tuple(float(x) for x in r["bbox"][0]),
                split=int(r["split"][0, 0]),
            )
        )
    return out


def load_metadata(data_root: Path) -> list[GolfDBAnno]:
    """Load the GolfDB annotation table from ``<data_root>/golfDB.mat``.

    Falls back to ``golfDB.pkl`` if pandas is available and the .mat is
    missing — but the .mat path is the canonical one and avoids the
    pandas dependency.
    """
    mat_path = data_root / "golfDB.mat"
    if mat_path.exists():
        return _load_metadata_mat(mat_path)

    pkl_path = data_root / "golfDB.pkl"
    if pkl_path.exists():
        try:
            import pandas as pd
        except ImportError as e:
            raise FileNotFoundError(
                f"golfDB.mat not found at {mat_path}; golfDB.pkl exists but "
                f"loading it requires pandas (not in core deps)."
            ) from e
        df = pd.read_pickle(pkl_path)
        return [
            GolfDBAnno(
                anno_id=int(row.id),
                youtube_id=str(row.youtube_id),
                player=str(row.player),
                sex=str(row.sex),
                club=str(row.club),
                view=str(row.view),
                slow=int(row.slow),
                events=tuple(int(x) for x in row.events),
                bbox=tuple(float(x) for x in row.bbox),
                split=int(row.split),
            )
            for row in df.itertuples()
        ]

    raise FileNotFoundError(
        f"GolfDB metadata not found in {data_root} (looked for golfDB.mat / golfDB.pkl). "
        "Pull from https://github.com/wmcnally/golfdb/tree/master/data."
    )


# ---------------------------------------------------------------------------
# Video locating + on-the-fly cropping
# ---------------------------------------------------------------------------

def _cropped_path(data_root: Path, anno: GolfDBAnno) -> Path:
    return data_root / "videos_160" / f"{anno.anno_id}.mp4"


def _source_path(data_root: Path, anno: GolfDBAnno) -> Path:
    # yt-dlp default output template tends to be <id>.<ext>; we accept any.
    direct = data_root / "videos" / f"{anno.youtube_id}.mp4"
    if direct.exists():
        return direct
    candidates = sorted((data_root / "videos").glob(f"{anno.youtube_id}.*"))
    return candidates[0] if candidates else direct


def _crop_and_slice(
    src: Path, anno: GolfDBAnno, *, target_dim: int = 360
) -> Path:
    """Apply the GolfDB bbox + event-window crop to a YouTube source video.

    Reproduces the official ``preprocess_videos.py`` shape (centered + padded
    to a square) but at a higher resolution than 160 so MediaPipe can land
    landmarks reliably. Writes to a temp file and returns the path; caller
    must delete.

    Frame count of the output = ``events[-1] - events[0] + 1`` (matches what
    the pre-built ``videos_160`` shards contain), so downstream phase index
    math works the same on either layout.
    """
    import cv2

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open: {src}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bx, by, bw, bh = anno.bbox
    x = int(src_w * bx)
    y = int(src_h * by)
    w = int(src_w * bw)
    h = int(src_h * bh)

    fd, tmp_str = tempfile.mkstemp(prefix=f"golfdb_{anno.anno_id}_", suffix=".mp4")
    os.close(fd)
    tmp_path = Path(tmp_str)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(tmp_path), fourcc, src_fps, (target_dim, target_dim))

    pad_color = (int(0.406 * 255), int(0.456 * 255), int(0.485 * 255))  # ImageNet mean (BGR)

    count = 0
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1  # GolfDB events are 1-indexed in the source
            if count < anno.events[0]:
                continue
            if count > anno.events[-1]:
                break
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            ch, cw = crop.shape[:2]
            ratio = target_dim / max(ch, cw)
            new_h = max(1, round(ch * ratio))
            new_w = max(1, round(cw * ratio))
            resized = cv2.resize(crop, (new_w, new_h))
            dh = target_dim - new_h
            dw = target_dim - new_w
            top, bot = dh // 2, dh - dh // 2
            left, right = dw // 2, dw - dw // 2
            framed = cv2.copyMakeBorder(
                resized, top, bot, left, right, cv2.BORDER_CONSTANT, value=pad_color
            )
            out.write(framed)
            written += 1
    finally:
        cap.release()
        out.release()

    if written == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"crop produced 0 frames for anno {anno.anno_id} "
            f"(events={anno.events[0]}..{anno.events[-1]}, source has fps={src_fps})"
        )
    return tmp_path


# ---------------------------------------------------------------------------
# Pose cache (for the live overlay; loaded by landmarks_for_entry)
# ---------------------------------------------------------------------------

def _pose_cache_dir(data_root: Path) -> Path:
    return data_root / "pose_cache"


def _save_pose_cache(
    data_root: Path,
    entry_id: int,
    landmarks: np.ndarray,
    fps: float,
    motion: str = "golf_swing",
) -> None:
    cache_dir = _pose_cache_dir(data_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_dir / f"{entry_id}.npz",
        landmarks=landmarks.astype(np.float32),
        fps=np.float32(fps),
        motion=np.array(motion, dtype="U32"),
    )


@lru_cache(maxsize=256)
def _load_pose_cache(path_str: str) -> tuple[np.ndarray, str, float]:
    npz = np.load(path_str, allow_pickle=False)
    return (
        np.asarray(npz["landmarks"], dtype=np.float32),
        str(npz["motion"].item()) if npz["motion"].shape == () else str(npz["motion"][()]),
        float(npz["fps"].item()),
    )


def _resolve_data_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    env = os.environ.get("MIMICCOACH_GOLFDB_ROOT")
    if env:
        return Path(env)
    return Path(__file__).parent / "data" / "golfdb"


def landmarks_for_entry(
    entry_id: int, *, data_root: Path | None = None
) -> tuple[np.ndarray, str, float] | None:
    """Reload landmarks for a previously-loaded golfdb manifest row.

    Mirrors ``reference.synthetic.landmarks_for_entry`` so the live
    ``/analyze`` handler can show a real pro skeleton overlay when the
    matched clip came from GolfDB. Returns None if no cache file is on
    disk for ``entry_id`` (e.g. running in a stripped Modal image).
    """
    if entry_id < ID_OFFSET:
        return None
    root = _resolve_data_root(data_root)
    path = _pose_cache_dir(root) / f"{entry_id}.npz"
    if not path.exists():
        return None
    return _load_pose_cache(str(path))


# ---------------------------------------------------------------------------
# Per-clip processing
# ---------------------------------------------------------------------------

def _phase_boundaries_from_events(
    events: tuple[int, ...], num_frames: int
) -> list[tuple[str, int, int]]:
    """Convert source-frame GolfDB events into cropped-clip phase windows.

    The cropped clip's frame 0 corresponds to source frame ``events[0]``
    (matches the official preprocess_videos.py output and our own
    ``_crop_and_slice``). So in cropped-clip frame numbering, the relevant
    frame index for event ``i`` is ``events[i] - events[0]``.

    Falls back to equal-partition for any phase whose event-derived window
    would be degenerate (start >= end). That keeps the row in the
    manifest even on edge-case annotations, at the cost of slightly less
    discriminative tokens for those phases.
    """
    base = events[0]
    rel = [max(0, e - base) for e in events]
    rel = [min(num_frames, r) for r in rel]

    out: list[tuple[str, int, int]] = []
    needs_fallback = False
    for name, (lo, hi) in zip(PHASE_NAMES, _PHASE_BOUNDS, strict=True):
        s, e = rel[lo], rel[hi]
        if e <= s:
            needs_fallback = True
            break
        out.append((name, s, e))

    if needs_fallback or len(out) != len(PHASE_NAMES):
        if num_frames < len(PHASE_NAMES):
            raise RuntimeError(
                f"clip too short ({num_frames}f) for {len(PHASE_NAMES)} phases"
            )
        edges = np.linspace(0, num_frames, len(PHASE_NAMES) + 1, dtype=np.int64)
        out = [
            (name, int(edges[i]), int(edges[i + 1]))
            for i, name in enumerate(PHASE_NAMES)
        ]
    return out


def _process_clip(
    video_path: Path,
    anno: GolfDBAnno,
    *,
    extractor: PoseExtractor,
    embedder: HandCraftedEmbedder,
    data_root: Path,
) -> dict[str, Any]:
    landmarks, meta = extractor.extract(video_path)
    if meta.detected_frames < max(8, meta.num_frames // 4):
        raise RuntimeError(
            f"pose detection too sparse ({meta.detected_frames}/{meta.num_frames} "
            f"frames) for anno {anno.anno_id}"
        )

    boundaries = _phase_boundaries_from_events(anno.events, meta.num_frames)
    per_frame = embedder.embed_frames(landmarks)
    tokens = [
        (name, phase_token(per_frame, s, e)) for (name, s, e) in boundaries
    ]
    token_array = np.stack([t[1] for t in tokens], axis=0)

    body_bucket = body_type_bucket(landmarks)

    entry_id = ID_OFFSET + anno.anno_id
    _save_pose_cache(data_root, entry_id, landmarks, fps=meta.fps)

    return {
        "id": entry_id,
        "phase_tokens": token_array.tolist(),
        "payload": {
            "sport": "golf",
            "motion": "golf_swing",
            "skill_level": "pro",
            "body_type": body_bucket,
            "athlete": _format_player(anno.player),
            "source": "golfdb",
            "source_url": f"https://www.youtube.com/watch?v={anno.youtube_id}",
            "license_note": "GolfDB CC BY-NC 4.0; pose-only retention",
            "club": anno.club,
            "view": anno.view,
            "sex": anno.sex,
            "slow_motion": bool(anno.slow),
            "split": anno.split,
        },
    }


def _format_player(name: str) -> str:
    """GolfDB stores names ALL CAPS. Title-case so the UI looks human."""
    return " ".join(part.capitalize() for part in name.split()) or name


# ---------------------------------------------------------------------------
# Public iter_rows
# ---------------------------------------------------------------------------

def iter_rows(
    data_root: Path | None = None,
    *,
    limit: int | None = None,
    only_anno_ids: set[int] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield manifest rows for every annotation we can process.

    "Can process" = metadata row + a video file on disk (either pre-cropped
    in ``videos_160/`` or raw source in ``videos/``). Annotations whose
    video is missing, unreadable, or where MediaPipe can't land enough
    landmarks are silently skipped — the pipeline is meant to gracefully
    handle partial downloads (the canonical ``videos_160.zip`` is 18 GB,
    so subsetting is the realistic flow).

    Args:
      data_root: GolfDB root with ``golfDB.mat`` + ``videos_160/`` and/or
        ``videos/``. Defaults to ``$MIMICCOACH_GOLFDB_ROOT`` or
        ``backend/reference/data/golfdb``.
      limit: cap the number of yielded rows.
      only_anno_ids: if set, restrict to these annotation ids (for tests
        and curated subsets).
    """
    root = _resolve_data_root(data_root)
    annos = load_metadata(root)
    if only_anno_ids is not None:
        annos = [a for a in annos if a.anno_id in only_anno_ids]

    extractor = PoseExtractor()
    embedder = HandCraftedEmbedder()

    yielded = 0
    skipped_missing = 0
    skipped_failed = 0
    try:
        for anno in annos:
            if limit is not None and yielded >= limit:
                break

            cropped = _cropped_path(root, anno)
            if cropped.exists():
                video_path = cropped
                tmp_to_cleanup: Path | None = None
            else:
                source = _source_path(root, anno)
                if not source.exists():
                    skipped_missing += 1
                    continue
                try:
                    video_path = _crop_and_slice(source, anno)
                    tmp_to_cleanup = video_path
                except Exception as e:  # log and skip — partial datasets are normal
                    logger.warning("crop failed for anno %s: %s", anno.anno_id, e)
                    skipped_failed += 1
                    continue

            try:
                # The MediaPipe Tasks API enforces monotonically-increasing
                # timestamps per PoseLandmarker instance. Each clip restarts
                # at t=0, so we tear down + reopen the landmarker between
                # clips. Model bundle stays cached on disk; the reopen is
                # ~100ms, dominated by clip pose extraction itself.
                extractor.close()
                row = _process_clip(
                    video_path,
                    anno,
                    extractor=extractor,
                    embedder=embedder,
                    data_root=root,
                )
            except Exception as e:  # log and skip — partial datasets are normal
                logger.warning("process failed for anno %s: %s", anno.anno_id, e)
                skipped_failed += 1
                continue
            finally:
                if tmp_to_cleanup is not None:
                    tmp_to_cleanup.unlink(missing_ok=True)

            yielded += 1
            yield row
    finally:
        extractor.close()

    logger.info(
        "golfdb loader: yielded=%d skipped_missing=%d skipped_failed=%d total_meta=%d",
        yielded,
        skipped_missing,
        skipped_failed,
        len(annos),
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="GolfDB loader smoke test")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional jsonl output path (one row per line)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    sink: io.StringIO | Any = (
        args.out.open("w") if args.out else io.StringIO()
    )
    n = 0
    try:
        for row in iter_rows(args.data_root, limit=args.limit):
            n += 1
            payload = row["payload"]
            print(
                f"[{n}] id={row['id']} {payload['athlete']:<22}"
                f" {payload['club']:<7} {payload['view']:<13}"
                f" body_type={payload['body_type']}"
                f" tokens={len(row['phase_tokens'])}x{len(row['phase_tokens'][0])}"
            )
            if args.out:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if args.out:
            sink.close()
    print(f"yielded {n} rows")


if __name__ == "__main__":
    _cli()
