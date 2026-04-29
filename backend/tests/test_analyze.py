"""End-to-end test of analyze_from_landmarks() against a synthetic clip.

We bypass the video extraction step (cv2 + mediapipe need a real MP4)
and feed pre-built MediaPipe-shaped landmarks straight into the pipeline.
This proves segment → embed → query → coach is wired correctly without
requiring real video data, which lands in P3.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import app
from pipeline.skeleton_map import MP_INDEX
from qdrant_io.client import make_client
from qdrant_io.schema import create_collection
from qdrant_io.upsert import manifest_to_points, upsert_points


def _coherent_serve_landmarks(T: int = 90) -> np.ndarray:
    """Synthetic 'tennis serve' clip: tossing arm rises, hitting arm peaks
    later, elbow flexes near trophy."""
    base = np.zeros((33, 4), dtype=np.float32)
    base[:, 3] = 0.9
    base[MP_INDEX["nose"]] = [0.50, 0.15, 0.0, 0.9]
    base[MP_INDEX["left_shoulder"]] = [0.42, 0.30, 0.0, 0.9]
    base[MP_INDEX["right_shoulder"]] = [0.58, 0.30, 0.0, 0.9]
    base[MP_INDEX["left_elbow"]] = [0.38, 0.45, 0.0, 0.9]
    base[MP_INDEX["right_elbow"]] = [0.62, 0.45, 0.0, 0.9]
    base[MP_INDEX["left_wrist"]] = [0.34, 0.58, 0.0, 0.9]
    base[MP_INDEX["right_wrist"]] = [0.66, 0.58, 0.0, 0.9]
    base[MP_INDEX["left_hip"]] = [0.45, 0.60, 0.0, 0.9]
    base[MP_INDEX["right_hip"]] = [0.55, 0.60, 0.0, 0.9]
    base[MP_INDEX["left_knee"]] = [0.45, 0.80, 0.0, 0.9]
    base[MP_INDEX["right_knee"]] = [0.55, 0.80, 0.0, 0.9]
    base[MP_INDEX["left_ankle"]] = [0.45, 0.98, 0.0, 0.9]
    base[MP_INDEX["right_ankle"]] = [0.55, 0.98, 0.0, 0.9]

    clip = np.tile(base, (T, 1, 1))
    t = np.arange(T)

    # Toss: left wrist rises (smaller y) early
    toss = np.where(
        (t >= int(T * 0.15)) & (t <= int(T * 0.45)),
        -0.18 * np.sin(np.pi * (t - int(T * 0.15)) / int(T * 0.30)),
        0.0,
    )
    clip[:, MP_INDEX["left_wrist"], 1] += toss

    # Trophy / contact: right wrist rises, peak around middle-late
    contact = np.where(
        (t >= int(T * 0.55)) & (t <= int(T * 0.90)),
        -0.25 * np.sin(np.pi * (t - int(T * 0.55)) / int(T * 0.35)),
        0.0,
    )
    clip[:, MP_INDEX["right_wrist"], 1] += contact

    # Trophy: pull right elbow up (it follows the wrist a little)
    elbow_lift = np.where(
        (t >= int(T * 0.50)) & (t <= int(T * 0.65)),
        -0.15 * np.sin(np.pi * (t - int(T * 0.50)) / int(T * 0.15)),
        0.0,
    )
    clip[:, MP_INDEX["right_elbow"], 1] += elbow_lift

    return clip


@pytest.fixture
def client_with_seeded_serve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Spin up a local Qdrant and seed it with one synthetic 'pro' tennis serve."""
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.delenv("QDRANT_URL", raising=False)

    client = make_client()
    create_collection(client, recreate=True)

    # Reuse our pipeline to derive the reference's phase_tokens deterministically.
    from pipeline.embed import HandCraftedEmbedder, phase_tokens
    from pipeline.segment import segment_video

    landmarks = _coherent_serve_landmarks(T=90)
    boundaries = segment_video(landmarks, motion="tennis_serve")
    per_frame = HandCraftedEmbedder().embed_frames(landmarks)
    tokens = phase_tokens(per_frame, boundaries)
    token_array = np.stack([t[1] for t in tokens], axis=0)

    upsert_points(client, manifest_to_points([
        {
            "id": 1,
            "phase_tokens": token_array.tolist(),
            "payload": {
                "sport": "tennis",
                "motion": "tennis_serve",
                "skill_level": "pro",
                "body_type": "balanced",
                "athlete": "Synthetic Pro",
                "source_url": "https://example.com/synthetic",
                "license_note": "synthetic-test-data",
            },
        },
    ]))
    yield client
    client.close()


def test_analyze_unknown_motion_returns_error(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "q"))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    landmarks = _coherent_serve_landmarks(T=60)
    result = app.analyze_from_landmarks(landmarks, motion="not_a_motion")
    assert "error" in result
    assert "supported" in result


def test_analyze_returns_match_for_seeded_serve(client_with_seeded_serve) -> None:
    landmarks = _coherent_serve_landmarks(T=90)
    result = app.analyze_from_landmarks(
        landmarks, motion="tennis_serve", client=client_with_seeded_serve
    )

    assert result["match"] is not None
    assert result["match"]["athlete"] == "Synthetic Pro"
    # Self-similar (we seeded the library with the same synthetic clip) → high score.
    assert result["match"]["score"] > 0.95

    # Per-phase scores have one entry per phase, in declared order.
    phases = [p["phase"] for p in result["per_phase_scores"]]
    assert phases == ["stance", "toss", "trophy", "contact", "follow_through"]
    for entry in result["per_phase_scores"]:
        assert 0.0 <= entry["score"] <= 1.0001

    # Coaching tip is non-empty and references the weakest phase.
    assert result["weakest_phase"] in phases
    assert result["coaching_tip"]
    assert isinstance(result["coaching_tip"], str)

    # User metadata is present.
    assert result["user"]["sport"] == "tennis"
    assert result["user"]["motion"] == "tennis_serve"
    assert result["user"]["body_type"] in {"narrow", "balanced", "broad"}
    assert len(result["user"]["phases"]) == 5

    # Filters round-trip into the response.
    assert result["filters_applied"]["sport"] == "tennis"
    assert result["filters_applied"]["motion"] == "tennis_serve"

    # JSON-serializable end-to-end (this is what the API returns over HTTP).
    json.dumps(result)


def test_analyze_no_match_when_filters_exclude_everything(client_with_seeded_serve) -> None:
    landmarks = _coherent_serve_landmarks(T=60)
    result = app.analyze_from_landmarks(
        landmarks,
        motion="tennis_serve",
        skill_level="beginner",  # only 'pro' was seeded
        client=client_with_seeded_serve,
    )
    assert result["match"] is None
    assert "error" in result
    assert result["coaching_tip"]  # graceful default


def test_analyze_skill_level_filter_returns_filtered_payload(
    client_with_seeded_serve,
) -> None:
    landmarks = _coherent_serve_landmarks(T=90)
    result = app.analyze_from_landmarks(
        landmarks,
        motion="tennis_serve",
        skill_level="pro",
        client=client_with_seeded_serve,
    )
    assert result["match"]["skill_level"] == "pro"
    assert result["filters_applied"]["skill_level"] == "pro"
