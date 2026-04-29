"""Unit tests for the embedding pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.embed import (
    EMBED_DIM,
    HandCraftedEmbedder,
    MotionBERTEmbedder,
    phase_token,
    phase_tokens,
)
from pipeline.skeleton_map import MP_INDEX


def _coherent_base_pose() -> np.ndarray:
    """A static standing pose with a sensible torso length, in normalized image space."""
    base = np.zeros((33, 4), dtype=np.float32)
    base[:, 3] = 0.9  # visibility
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
    return base


def _synthetic_landmarks(T: int = 60, seed: int = 0) -> np.ndarray:
    """Smooth synthetic MediaPipe landmarks: a slow drift + small jitter."""
    rng = np.random.default_rng(seed)
    # Pose around a plausible body shape (normalized image space).
    base = rng.uniform(0.3, 0.7, size=(33, 4)).astype(np.float32)
    base[:, 3] = 0.9  # visibility
    drift = np.linspace(0, 0.05, T, dtype=np.float32)[:, None, None]
    jitter = rng.normal(0, 0.005, size=(T, 33, 4)).astype(np.float32)
    return base[None, :, :] + drift + jitter


def test_handcrafted_embedder_shape_and_finite() -> None:
    landmarks = _synthetic_landmarks(T=60)
    emb = HandCraftedEmbedder().embed_frames(landmarks)
    assert emb.shape == (60, EMBED_DIM)
    assert np.isfinite(emb).all()


def test_handcrafted_embedder_is_deterministic() -> None:
    landmarks = _synthetic_landmarks(T=30)
    a = HandCraftedEmbedder(seed=42).embed_frames(landmarks)
    b = HandCraftedEmbedder(seed=42).embed_frames(landmarks)
    np.testing.assert_array_equal(a, b)


def test_handcrafted_embedder_different_seeds_differ() -> None:
    landmarks = _synthetic_landmarks(T=30)
    a = HandCraftedEmbedder(seed=1).embed_frames(landmarks)
    b = HandCraftedEmbedder(seed=2).embed_frames(landmarks)
    assert not np.allclose(a, b)


def test_handcrafted_embedder_distinguishes_motions() -> None:
    """Two distinct motion patterns (arm swing vs leg lift) should produce
    different per-frame embeddings at peak motion."""
    base = _coherent_base_pose()

    T = 60
    swing = np.sin(np.linspace(0, 4 * np.pi, T)) * 0.15

    # Motion A: arms raise vertically
    a = np.tile(base, (T, 1, 1))
    a[:, MP_INDEX["left_wrist"], 1] -= swing
    a[:, MP_INDEX["right_wrist"], 1] -= swing
    a[:, MP_INDEX["left_elbow"], 1] -= swing * 0.5
    a[:, MP_INDEX["right_elbow"], 1] -= swing * 0.5

    # Motion B: knee drive (legs lift)
    b = np.tile(base, (T, 1, 1))
    b[:, MP_INDEX["left_knee"], 1] -= swing
    b[:, MP_INDEX["right_knee"], 1] -= swing
    b[:, MP_INDEX["left_ankle"], 1] -= swing
    b[:, MP_INDEX["right_ankle"], 1] -= swing

    embedder = HandCraftedEmbedder()
    ea = embedder.embed_frames(a)
    eb = embedder.embed_frames(b)

    # Per-frame cosine similarity between the two motion sequences.
    norm_a = np.linalg.norm(ea, axis=1) + 1e-8
    norm_b = np.linalg.norm(eb, axis=1) + 1e-8
    cos = (ea * eb).sum(axis=1) / (norm_a * norm_b)

    # At motion peaks the embeddings should diverge — require min cos < 0.95.
    assert cos.min() < 0.95, (
        f"distinct motions produced near-identical embeddings "
        f"(min cos={cos.min():.4f}, max cos={cos.max():.4f})"
    )


def test_phase_token_is_l2_unit() -> None:
    rng = np.random.default_rng(0)
    per_frame = rng.standard_normal((40, EMBED_DIM)).astype(np.float32)
    tok = phase_token(per_frame, 0, 10)
    assert tok.shape == (EMBED_DIM,)
    np.testing.assert_almost_equal(np.linalg.norm(tok), 1.0, decimal=5)


def test_phase_token_rejects_bad_window() -> None:
    per_frame = np.zeros((10, EMBED_DIM), dtype=np.float32)
    with pytest.raises(ValueError):
        phase_token(per_frame, 5, 5)
    with pytest.raises(ValueError):
        phase_token(per_frame, -1, 4)
    with pytest.raises(ValueError):
        phase_token(per_frame, 0, 11)


def test_phase_tokens_returns_one_per_window() -> None:
    rng = np.random.default_rng(0)
    per_frame = rng.standard_normal((50, EMBED_DIM)).astype(np.float32)
    boundaries = [
        ("setup", 0, 10),
        ("descent", 10, 25),
        ("bottom", 25, 30),
        ("ascent", 30, 45),
        ("lockout", 45, 50),
    ]
    tokens = phase_tokens(per_frame, boundaries)
    assert [name for name, _ in tokens] == [b[0] for b in boundaries]
    for _, vec in tokens:
        np.testing.assert_almost_equal(np.linalg.norm(vec), 1.0, decimal=5)


def test_motionbert_stub_raises() -> None:
    landmarks = _synthetic_landmarks(T=10)
    with pytest.raises(NotImplementedError):
        MotionBERTEmbedder().embed_frames(landmarks)


def test_handcrafted_embedder_handles_visibility_channel() -> None:
    """Both (T, 33, 3) and (T, 33, 4) inputs should work."""
    rng = np.random.default_rng(0)
    landmarks_4 = rng.uniform(0, 1, size=(20, 33, 4)).astype(np.float32)
    landmarks_3 = landmarks_4[..., :3]
    embedder = HandCraftedEmbedder()
    e4 = embedder.embed_frames(landmarks_4)
    e3 = embedder.embed_frames(landmarks_3)
    np.testing.assert_array_almost_equal(e3, e4)
