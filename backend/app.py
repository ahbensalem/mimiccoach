"""MimicCoach Modal app — full /analyze pipeline.

The Modal app exposes one FastAPI ASGI app with three routes:

  GET  /healthz   - liveness probe (used by warm_modal.sh cron)
  GET  /motions   - list supported motions and their phases
  POST /analyze   - multipart MP4 + form fields → JSON match + scores

The /analyze handler delegates to `analyze_from_landmarks()`, which is
also unit-testable in isolation using synthetic landmarks. The HTTP
wrapper just owns video I/O and pose extraction.
"""
from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

import modal

# FastAPI types must be importable at module scope so `from __future__ import
# annotations` doesn't trip pydantic's forward-reference resolution on the
# /analyze handler's `video: UploadFile` parameter. Importing them here is
# safe both locally (fastapi is in the dev venv) and inside the Modal image
# (fastapi[standard] is pip_installed in `image` below).
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")  # mediapipe + opencv runtime
    .pip_install(
        "mediapipe>=0.10.18",
        "numpy>=1.26,<2.2",
        "opencv-python-headless>=4.10",
        "torch>=2.4",
        "qdrant-client>=1.12",
        "pydantic>=2.9",
        "pyyaml>=6.0",
        "fastapi[standard]>=0.115",
        "python-multipart>=0.0.12",
        "scipy>=1.13",
    )
    .add_local_python_source("pipeline", "qdrant_io")
    # add_local_python_source ships .py files only; motions.yaml is a sibling
    # data file segment.py reads at request time, so it has to be added
    # explicitly. Lands at /root/pipeline/ to match the Python source layout.
    .add_local_file(
        str(Path(__file__).parent / "pipeline" / "motions.yaml"),
        "/root/pipeline/motions.yaml",
    )
)

app = modal.App("mimiccoach", image=image)


# ---------------------------------------------------------------------------
# Pipeline glue (testable without a real video)
# ---------------------------------------------------------------------------

def analyze_from_landmarks(
    landmarks,
    *,
    motion: str,
    skill_level: str | None = None,
    body_type_override: str | None = None,
    limit: int = 5,
    pose_meta: dict[str, Any] | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Run segment → embed → query → coach against pre-extracted landmarks.

    Returns the same JSON shape the /analyze HTTP endpoint produces. Pass
    `client` to reuse an existing Qdrant client (tests do this; the embedded
    on-disk mode allows only one open client per path).
    """
    import numpy as np

    from pipeline.body_type import body_type_bucket
    from pipeline.coach import coach_from_per_phase
    from pipeline.embed import HandCraftedEmbedder, phase_tokens
    from pipeline.segment import _config, phase_names, segment_video
    from qdrant_io.client import make_client
    from qdrant_io.query import query_motions

    motions_cfg = _config()["motions"]
    if motion not in motions_cfg:
        return {
            "error": f"unknown motion: {motion!r}",
            "supported": sorted(motions_cfg),
        }
    sport = motions_cfg[motion]["sport"]

    # 1. Segment phases
    boundaries = segment_video(landmarks, motion=motion)
    names = phase_names(motion)

    # 2. Embed
    embedder = HandCraftedEmbedder()
    per_frame = embedder.embed_frames(landmarks)
    tokens = phase_tokens(per_frame, boundaries)
    token_array = np.stack([t[1] for t in tokens], axis=0)

    # 3. Auto-derive body type if not overridden
    derived_body_type = body_type_bucket(landmarks)

    # 4. Query Qdrant.
    # body_type is only applied if explicitly requested; the auto-derived value
    # is returned for the UI but not silently used to constrain results.
    qd_client = client if client is not None else make_client()
    matches = query_motions(
        qd_client,
        token_array,
        sport=sport,
        motion=motion,
        skill_level=skill_level,
        body_type=body_type_override,
        limit=limit,
    )

    response: dict[str, Any] = {
        "user": {
            "sport": sport,
            "motion": motion,
            "body_type": derived_body_type,
            "phases": [
                {"name": name, "start_frame": int(s), "end_frame": int(e)}
                for (name, s, e) in boundaries
            ],
            **(pose_meta or {}),
        },
        "filters_applied": {
            "sport": sport,
            "motion": motion,
            "skill_level": skill_level,
            "body_type": body_type_override,
        },
    }

    if not matches:
        response["error"] = (
            "no reference clips matched the filters — try widening "
            "the filters or check the reference library has been built"
        )
        response["match"] = None
        response["per_phase_scores"] = [
            {"phase": n, "score": 0.0} for n in names
        ]
        response["weakest_phase"] = names[0]
        response["coaching_tip"] = (
            "No reference clip available for these filters yet."
        )
        response["alternatives"] = []
        return response

    top = matches[0]
    coaching = coach_from_per_phase(motion, top.per_phase_scores, names)

    response["match"] = {
        "point_id": str(top.point_id),
        "score": top.score,
        "athlete": top.payload.get("athlete"),
        "source_url": top.payload.get("source_url"),
        "skill_level": top.payload.get("skill_level"),
        "body_type": top.payload.get("body_type"),
        "pose_url": top.payload.get("pose_url"),
        "video_url": top.payload.get("video_url"),
    }
    response["per_phase_scores"] = [
        {"phase": name, "score": float(score)}
        for name, score in zip(names, top.per_phase_scores, strict=True)
    ]
    response["weakest_phase"] = coaching.weakest_phase
    response["coaching_tip"] = coaching.tip
    response["alternatives"] = [
        {
            "point_id": str(m.point_id),
            "score": m.score,
            "athlete": m.payload.get("athlete"),
            "skill_level": m.payload.get("skill_level"),
        }
        for m in matches[1:]
    ]
    return response


def landmarks_to_pose_payload(landmarks) -> list[list[list[float]]]:
    """Compact pose JSON for the frontend overlay: (T, 33, 4) → list-of-lists.

    We keep all 33 MediaPipe landmarks so the frontend can draw the full
    skeleton; visibility is preserved so the canvas can fade out occluded
    joints.
    """
    arr = landmarks.tolist()
    return arr  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@app.function(schedule=modal.Period(minutes=4), secrets=[modal.Secret.from_name("qdrant"), modal.Secret.from_name("mimiccoach-warm")])
def warm_keep() -> None:
    """Periodic no-op so the analyze container stays warm during demo windows.
    Modal scales analyze containers to zero on inactivity; this scheduled
    function ensures at least one is up. Cheap (a few CPU-seconds every 4
    min) and avoids 30s cold-starts during live judging."""
    import contextlib
    import os as _os
    import urllib.error
    import urllib.request

    base = _os.environ.get("MIMICCOACH_PUBLIC_URL")
    if not base:
        # No public URL yet — silent no-op (the schedule still runs but does
        # nothing actionable). Set MIMICCOACH_PUBLIC_URL in the Modal app's
        # env once `modal deploy` returns the asgi URL.
        return
    # Best-effort: a transient blip shouldn't keep the schedule from firing
    # again in 4 minutes.
    with contextlib.suppress(urllib.error.URLError):
        urllib.request.urlopen(f"{base}/healthz", timeout=15).read(64)


@app.function(timeout=300, max_containers=4, secrets=[modal.Secret.from_name("qdrant")])
@modal.asgi_app()
def fastapi_app():
    web = FastAPI(title="MimicCoach")

    web.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("FRONTEND_ORIGIN", "*").split(","),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @web.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "service": "mimiccoach"}

    @web.get("/motions")
    async def list_motions() -> dict[str, list[dict[str, Any]]]:
        from pipeline.segment import _config

        cfg = _config()["motions"]
        return {
            "motions": [
                {
                    "key": key,
                    "sport": v["sport"],
                    "label": v["label"],
                    "phases": [p["name"] for p in v["phases"]],
                    "is_hero": bool(v.get("is_hero", False)),
                }
                for key, v in cfg.items()
            ]
        }

    @web.post("/analyze")
    async def analyze(
        video: UploadFile = File(...),
        motion: str = Form(...),
        skill_level: str | None = Form(None),
        body_type: str | None = Form(None),
        limit: int = Form(5),
    ) -> dict[str, Any]:
        if video.content_type not in {"video/mp4", "video/quicktime", "application/octet-stream"}:
            raise HTTPException(415, f"unsupported video type: {video.content_type}")

        # Persist upload to a tmp file (cv2 + mediapipe need a path).
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(await video.read())
            tmp_path = Path(tmp.name)

        try:
            from pipeline.pose_extract import PoseExtractor

            extractor = PoseExtractor()
            try:
                landmarks, meta = extractor.extract(tmp_path)
            finally:
                extractor.close()

            result = analyze_from_landmarks(
                landmarks,
                motion=motion,
                skill_level=skill_level,
                body_type_override=body_type,
                limit=limit,
                pose_meta={
                    "fps": meta.fps,
                    "num_frames": meta.num_frames,
                    "detected_frames": meta.detected_frames,
                    "width": meta.width,
                    "height": meta.height,
                },
            )
            # Inline the user's pose JSON for client-side skeleton overlay.
            result["user"]["pose"] = landmarks_to_pose_payload(landmarks)
            return result
        finally:
            with contextlib.suppress(OSError):
                tmp_path.unlink()

    return web


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Run with:")
    print("  modal serve backend/app.py    # local dev (hot-reload)")
    print("  modal deploy backend/app.py   # production")
