"""Per-frame and per-phase pose embeddings.

Architecture: a `PhaseEmbedder` protocol with two concrete implementations.

  * `HandCraftedEmbedder` — joint angles + velocities + a fixed (seeded)
    Gaussian projection to 512 dims. Deterministic, fast, no network deps.
    This is the **primary** implementation for the hackathon MVP because
    integration speed and demo reliability matter more than SOTA quality.

  * `MotionBERTEmbedder` — stub for the SOTA path. Documented swap-in once
    the rest of the pipeline is end-to-end. Locked in the plan as the
    eventual primary; raises NotImplementedError today.

Both produce per-frame `(T, 512)` arrays. `phase_token()` mean-pools an
embedding window over a phase and L2-normalizes the result, yielding the
single 512-d token that becomes one element of a Qdrant multivector point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .skeleton_map import H36M_INDEX, mediapipe_to_h36m

EMBED_DIM: int = 512


class PhaseEmbedder(Protocol):
    """Maps a sequence of MediaPipe-33 landmarks to per-frame 512-d vectors."""

    def embed_frames(self, mp_landmarks: np.ndarray) -> np.ndarray:
        """Args: (T, 33, 4) MediaPipe landmarks (x, y, z, visibility).

        Returns: (T, 512) per-frame embeddings.
        """
        ...


# ---------------------------------------------------------------------------
# Hand-crafted embedder (primary for MVP)
# ---------------------------------------------------------------------------

# Pairs of (joint, parent, child) used to compute joint angles in H36M-17 space.
# Indexed via H36M_INDEX (see skeleton_map.py).
_ANGLE_TRIPLES: tuple[tuple[str, str, str], ...] = (
    # arms
    ("left_elbow", "left_shoulder", "left_wrist"),
    ("right_elbow", "right_shoulder", "right_wrist"),
    ("left_shoulder", "thorax", "left_elbow"),
    ("right_shoulder", "thorax", "right_elbow"),
    # legs
    ("left_knee", "left_hip", "left_ankle"),
    ("right_knee", "right_hip", "right_ankle"),
    ("left_hip", "pelvis", "left_knee"),
    ("right_hip", "pelvis", "right_knee"),
    # spine
    ("spine", "pelvis", "thorax"),
)


def _angle_at(joint: np.ndarray, parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    """Compute the angle at `joint` between vectors (parent-joint) and (child-joint).

    Args: each is (T, 3). Returns: (T,) in radians, in [0, π].
    """
    a = parent - joint
    b = child - joint
    na = np.linalg.norm(a, axis=-1) + 1e-8
    nb = np.linalg.norm(b, axis=-1) + 1e-8
    cos = np.einsum("ti,ti->t", a, b) / (na * nb)
    return np.arccos(np.clip(cos, -1.0, 1.0))


_KEEP_NAMES: tuple[str, ...] = (
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "thorax",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
)


@dataclass
class HandCraftedEmbedder:
    """Pose-and-motion features projected to 512 dims via a fixed random basis.

    Per-frame raw feature vector (motion-centric — absolute body shape is
    intentionally dropped so retrieval focuses on what the body is *doing*):

      * 13 joints × 2 in-plane offsets (relative to pelvis, scaled by clip
        median torso length)             → 26 features
      * 9 joint angles in radians [0, π]  → 9 features
      * 13 joints × 2 in-plane velocities (Δoffset × 30, ~per-second) → 26
      * 9 angular velocities (Δangle × 30) → 9
        Total raw dim = 70

    Why 2D offsets rather than 3D: MediaPipe's z is depth-relative-to-hips
    and noisy on monocular video, especially for amateur uploads. The in-
    plane (image-space) signal is far more reliable, and the multi-view
    nature of the reference library (clips from different camera angles)
    means depth would alias retrieval more than help it.

    The output is L2-normalized per frame so cosine similarity in Qdrant
    only sees structural (not magnitude) information.

    Projection: a fixed (seeded) Gaussian random matrix W ∈ ℝ^{70 × 512},
    1/√d scaled. Random projection preserves pairwise distances
    (Johnson–Lindenstrauss).
    """

    seed: int = 1729
    velocity_fps: float = 30.0
    """Multiplier applied to per-frame deltas to put them on a comparable
    scale with positions/angles. Treat the raw frame rate as 30 fps; the
    actual fps is normalized away by `frame_skip` in pose_extract.PoseExtractor."""

    _proj: np.ndarray | None = None

    _RAW_DIM: int = 13 * 2 + len(_ANGLE_TRIPLES) + 13 * 2 + len(_ANGLE_TRIPLES)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        proj = rng.standard_normal((self._RAW_DIM, EMBED_DIM)).astype(np.float32)
        proj /= np.sqrt(self._RAW_DIM)
        self._proj = proj

    def embed_frames(self, mp_landmarks: np.ndarray) -> np.ndarray:
        if mp_landmarks.ndim != 3 or mp_landmarks.shape[1] != 33:
            raise ValueError(f"expected (T, 33, k); got {mp_landmarks.shape}")
        T = mp_landmarks.shape[0]

        # MediaPipe → H36M (drop visibility / extra channels).
        h36m = mediapipe_to_h36m(mp_landmarks[..., :3].astype(np.float32))  # (T, 17, 3)

        # Single torso length per clip (median across frames) to avoid per-
        # frame noise blowing up the normalization.
        pelvis = h36m[:, H36M_INDEX["pelvis"]]  # (T, 3)
        thorax = h36m[:, H36M_INDEX["thorax"]]  # (T, 3)
        torso_len = float(np.median(np.linalg.norm(thorax - pelvis, axis=-1))) + 1e-6

        # In-plane (x, y) offsets relative to pelvis, normalized by torso.
        keep_idx = np.array([H36M_INDEX[n] for n in _KEEP_NAMES], dtype=np.int64)
        offsets = (h36m[:, keep_idx, :2] - pelvis[:, None, :2]) / torso_len  # (T, 13, 2)

        # Joint angles in raw H36M coordinates (angles are scale-invariant).
        angles = np.stack(
            [
                _angle_at(
                    h36m[:, H36M_INDEX[j]],
                    h36m[:, H36M_INDEX[p]],
                    h36m[:, H36M_INDEX[c]],
                )
                for (j, p, c) in _ANGLE_TRIPLES
            ],
            axis=1,
        )  # (T, 9)

        # Velocities — scaled to ~per-second magnitude so they contribute
        # comparably to positions/angles in the random projection.
        vel = np.zeros_like(offsets)
        vel[1:] = (offsets[1:] - offsets[:-1]) * self.velocity_fps
        ang_vel = np.zeros_like(angles)
        ang_vel[1:] = (angles[1:] - angles[:-1]) * self.velocity_fps

        raw = np.concatenate(
            [
                offsets.reshape(T, -1),  # 26
                angles,                  # 9
                vel.reshape(T, -1),      # 26
                ang_vel,                 # 9
            ],
            axis=1,
        ).astype(np.float32)  # (T, 70)

        assert self._proj is not None
        emb = raw @ self._proj  # (T, 512)

        # L2-normalize each frame so cosine similarity sees structure only.
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        return emb / norms


# ---------------------------------------------------------------------------
# MotionBERT embedder (stretch / swap-in)
# ---------------------------------------------------------------------------

@dataclass
class MotionBERTEmbedder:
    """Stub for the MotionBERT swap-in.

    Integration TODOs (P5+ once HandCraftedEmbedder has carried us through
    the demo):
      1. `huggingface_hub.snapshot_download("walterzhu/MotionBERT")` to grab
         the checkpoint.
      2. Vendor the minimal subset of the MotionBERT model code from
         github.com/Walter0807/MotionBERT (DSTformer with ~16 layers).
      3. Map MediaPipe-33 → H36M-17 via `mediapipe_to_h36m()`, normalize
         to MotionBERT's expected scale (root-relative, hip-centered).
      4. Forward through the encoder, take the per-frame feature output
         (`(T, 17, 512)`), mean over the joint axis → `(T, 512)`.
      5. L2-normalize per frame for stable cosine similarity.

    Until then, calling this raises NotImplementedError so misuse is loud.
    """
    ckpt: str = "walterzhu/MotionBERT"
    device: str = "cpu"

    def embed_frames(self, mp_landmarks: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "MotionBERTEmbedder is a planned swap-in; HandCraftedEmbedder is the MVP path."
        )


# ---------------------------------------------------------------------------
# Per-phase pooling
# ---------------------------------------------------------------------------

def phase_token(per_frame: np.ndarray, start: int, end: int) -> np.ndarray:
    """Mean-pool per-frame embeddings over [start, end) and L2-normalize.

    Args:
        per_frame: (T, 512) embeddings.
        start, end: phase boundary frame indices, with start < end.

    Returns:
        (512,) L2-unit vector — the single token representing this phase.
    """
    if per_frame.ndim != 2 or per_frame.shape[1] != EMBED_DIM:
        raise ValueError(f"expected (T, {EMBED_DIM}); got {per_frame.shape}")
    if not (0 <= start < end <= per_frame.shape[0]):
        raise ValueError(
            f"phase window out of range: [{start}, {end}) for T={per_frame.shape[0]}"
        )
    pooled = per_frame[start:end].mean(axis=0)
    norm = np.linalg.norm(pooled) + 1e-8
    return (pooled / norm).astype(np.float32)


def phase_tokens(
    per_frame: np.ndarray,
    boundaries: list[tuple[str, int, int]],
) -> list[tuple[str, np.ndarray]]:
    """Apply `phase_token` to each (name, start, end) tuple."""
    return [(name, phase_token(per_frame, s, e)) for (name, s, e) in boundaries]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    """`python -m pipeline.embed sample.mp4 --motion tennis_serve` smoke test."""
    import argparse

    parser = argparse.ArgumentParser(description="MimicCoach embedding smoke test")
    parser.add_argument("video", help="Path to an MP4 clip")
    parser.add_argument(
        "--motion",
        default="tennis_serve",
        help="Motion key (used for phase boundaries)",
    )
    parser.add_argument(
        "--equal-partition",
        action="store_true",
        help="Skip segmentation; partition the video into equal-length phases.",
    )
    args = parser.parse_args()

    from pathlib import Path

    from .pose_extract import PoseExtractor

    extractor = PoseExtractor()
    landmarks, meta = extractor.extract(Path(args.video))
    print(f"extracted {meta.num_frames} frames @ {meta.fps:.1f} fps "
          f"({meta.detected_frames}/{meta.num_frames} detected)")

    embedder = HandCraftedEmbedder()
    per_frame = embedder.embed_frames(landmarks)
    print(f"per-frame embeddings: {per_frame.shape}")

    if args.equal_partition:
        # 5 equal-duration phases when no segmenter has been wired yet.
        T = per_frame.shape[0]
        names = ["p1", "p2", "p3", "p4", "p5"]
        edges = np.linspace(0, T, len(names) + 1, dtype=int)
        boundaries = [
            (n, int(edges[i]), int(edges[i + 1]))
            for i, n in enumerate(names)
        ]
    else:
        from .segment import segment_video  # lazy import (lands in P2)

        boundaries = segment_video(landmarks, motion=args.motion)

    tokens = phase_tokens(per_frame, boundaries)
    for name, vec in tokens:
        print(f"  {name:>20s}: dim={vec.shape[0]} norm={np.linalg.norm(vec):.4f}")


if __name__ == "__main__":
    _cli()
