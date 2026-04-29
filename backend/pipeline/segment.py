"""Phase segmentation for sport motions.

Two-stage design:

1. **Active-window detection** (motion-agnostic): compute total joint motion
   intensity per frame, smooth it, and trim leading/trailing low-motion
   frames. The active window is when the athlete is actually doing the
   motion — the segmenter never proposes phase boundaries outside it.

2. **Phase splitting** (per-motion): inside the active window, propose
   N phase boundaries.

   * Default: equal partition (works for any motion; consistent between
     user uploads and reference clips, which is what retrieval needs).
   * Override per motion: signal-based rules using key-joint velocities
     and angles (currently implemented for tennis serve as hero motion;
     other motions fall back to equal partition until tuned in P7).

The output is `list[(phase_name, start_frame, end_frame)]` consumed by
`pipeline.embed.phase_tokens()`.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .skeleton_map import MP_INDEX

CONFIG_PATH = Path(__file__).parent / "motions.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _config() -> dict[str, Any]:
    import yaml

    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def phase_names(motion: str) -> list[str]:
    cfg = _config()["motions"]
    if motion not in cfg:
        raise KeyError(f"unknown motion: {motion!r} (known: {list(cfg)})")
    return [p["name"] for p in cfg[motion]["phases"]]


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def smooth_1d(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered moving average. `window` ≥ 1; returns same shape as `signal`."""
    if window <= 1 or signal.shape[0] <= 1:
        return signal.astype(np.float32)
    w = min(window, signal.shape[0])
    if w % 2 == 0:
        w -= 1
    if w < 1:
        return signal.astype(np.float32)
    pad = w // 2
    kernel = np.ones(w, dtype=np.float32) / w
    padded = np.pad(signal.astype(np.float32), pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: signal.shape[0]]


def per_frame_velocity_magnitude(landmarks: np.ndarray) -> np.ndarray:
    """Sum of joint-velocity magnitudes per frame (T,) for MediaPipe-33 input.

    Used for active-window detection. Heavy joints (head landmarks, fingers)
    are excluded so that camera shake or noisy face landmarks don't dominate.
    """
    if landmarks.ndim != 3 or landmarks.shape[1] != 33:
        raise ValueError(f"expected (T, 33, k); got {landmarks.shape}")

    body_joints = (
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    )
    idx = np.array([MP_INDEX[n] for n in body_joints], dtype=np.int64)
    pos = landmarks[:, idx, :2].astype(np.float32)  # in-plane only

    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    mag = np.linalg.norm(vel, axis=-1)  # (T, 12)
    return mag.sum(axis=1)  # (T,)


def active_window(
    landmarks: np.ndarray,
    threshold_ratio: float = 0.15,
    min_active_frames: int = 4,
) -> tuple[int, int]:
    """Find the [start, end) window where the athlete is in motion.

    Threshold: a fraction of the peak smoothed motion intensity. Frames whose
    intensity falls below `threshold_ratio × peak` at the head/tail of the
    clip are trimmed.
    """
    T = landmarks.shape[0]
    if T <= min_active_frames:
        return (0, T)

    intensity = smooth_1d(per_frame_velocity_magnitude(landmarks), window=5)
    peak = float(intensity.max())
    if peak < 1e-6:
        return (0, T)

    threshold = peak * threshold_ratio
    above = intensity >= threshold
    # First and last True index.
    if not above.any():
        return (0, T)
    start = int(np.argmax(above))
    end = int(T - np.argmax(above[::-1]))

    # Guard: keep at least min_active_frames of headroom.
    if end - start < min_active_frames:
        return (0, T)
    return (start, end)


def equal_partition(
    start: int, end: int, n_phases: int
) -> list[tuple[int, int]]:
    """Split [start, end) into n_phases approximately-equal contiguous windows."""
    if n_phases < 1:
        raise ValueError("n_phases must be >= 1")
    if end - start < n_phases:
        raise ValueError(f"window {end - start} too small for {n_phases} phases")
    edges = np.linspace(start, end, n_phases + 1, dtype=np.int64)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_phases)]


# ---------------------------------------------------------------------------
# Per-motion segmenters
# ---------------------------------------------------------------------------

Segmenter = Callable[[np.ndarray, int, int], list[tuple[int, int]]]
"""(landmarks, active_start, active_end) -> list of (start, end) per phase."""


def _equal_segmenter(n_phases: int) -> Segmenter:
    def _impl(_landmarks: np.ndarray, start: int, end: int) -> list[tuple[int, int]]:
        return equal_partition(start, end, n_phases)
    return _impl


def _segment_tennis_serve(
    landmarks: np.ndarray, start: int, end: int, *, right_handed: bool = True
) -> list[tuple[int, int]]:
    """Signal-based 5-phase segmentation of a tennis serve.

    Phases (right-handed convention):
      stance · toss · trophy · contact · follow_through

    Heuristics:
      * toss boundary  = peak height of *non-dominant* wrist (ball release)
      * trophy boundary = max flex of *dominant* elbow (elbow angle minimum)
      * contact boundary = peak height of *dominant* wrist
      * stance/follow_through bracket the active window.

    Falls back to equal partition if the detected events don't strictly
    increase in time.
    """
    tossing = "left_wrist" if right_handed else "right_wrist"
    hitting = "right_wrist" if right_handed else "left_wrist"
    elbow = "right_elbow" if right_handed else "left_elbow"
    shoulder = "right_shoulder" if right_handed else "left_shoulder"

    win = landmarks[start:end, :, :3].astype(np.float32)
    if win.shape[0] < 5:
        return equal_partition(start, end, 5)

    # In MediaPipe normalized image space, smaller y means higher up. We
    # negate so "peak height" is "argmax of the signal".
    toss_y = -smooth_1d(win[:, MP_INDEX[tossing], 1], window=5)
    hit_y = -smooth_1d(win[:, MP_INDEX[hitting], 1], window=5)
    f_toss_peak = int(np.argmax(toss_y))
    f_hit_peak = int(np.argmax(hit_y))

    # Elbow angle at the dominant arm (shoulder-elbow-wrist).
    a = win[:, MP_INDEX[shoulder]] - win[:, MP_INDEX[elbow]]
    b = win[:, MP_INDEX[hitting]] - win[:, MP_INDEX[elbow]]
    na = np.linalg.norm(a, axis=-1) + 1e-8
    nb = np.linalg.norm(b, axis=-1) + 1e-8
    cos = np.einsum("ti,ti->t", a, b) / (na * nb)
    elbow_angle = smooth_1d(np.arccos(np.clip(cos, -1, 1)), window=5)
    f_trophy = int(np.argmin(elbow_angle))

    # Move active-window-relative indices back to absolute frame indices.
    f_toss_peak_abs = start + f_toss_peak
    f_trophy_abs = start + f_trophy
    f_hit_peak_abs = start + f_hit_peak

    # Strictly increasing? If not, fall back.
    if not (start < f_toss_peak_abs < f_trophy_abs < f_hit_peak_abs < end):
        return equal_partition(start, end, 5)

    # Inject 5–10% headroom so the "stance" phase isn't degenerate.
    stance_end = max(start + 1, start + (f_toss_peak_abs - start) // 4)

    return [
        (start, stance_end),                # stance
        (stance_end, f_toss_peak_abs),      # toss
        (f_toss_peak_abs, f_trophy_abs),    # trophy
        (f_trophy_abs, f_hit_peak_abs),     # contact
        (f_hit_peak_abs, end),              # follow_through
    ]


_MOTION_SEGMENTERS: dict[str, Segmenter] = {
    "tennis_serve": _segment_tennis_serve,
    # All other motions use equal partition until tuned in P7.
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_video(
    landmarks: np.ndarray,
    motion: str,
) -> list[tuple[str, int, int]]:
    """Segment a clip into phases for the given motion.

    Args:
        landmarks: (T, 33, k) MediaPipe landmarks.
        motion: motion key (e.g., "tennis_serve") matching motions.yaml.

    Returns:
        list of (phase_name, start_frame, end_frame). Frame indices are in
        the original clip's frame numbering. Always returns one entry per
        phase declared in motions.yaml (no missing or extra phases).
    """
    names = phase_names(motion)
    n = len(names)
    T = landmarks.shape[0]

    start, end = active_window(landmarks)

    # Make sure the active window has enough frames to host n phases.
    if end - start < n:
        start, end = 0, T
    if end - start < n:
        # Pathologically short clip — pad with degenerate phases so callers
        # can still run, but every phase ends up with at least 1 frame if we
        # fall back to single-frame windows.
        if T < n:
            raise ValueError(f"clip too short ({T} frames) for {n} phases")
        start, end = 0, T

    seg = _MOTION_SEGMENTERS.get(motion, _equal_segmenter(n))
    windows = seg(landmarks, start, end)
    if len(windows) != n:
        raise RuntimeError(
            f"segmenter for {motion!r} returned {len(windows)} windows; expected {n}"
        )

    return [(name, s, e) for name, (s, e) in zip(names, windows, strict=True)]
