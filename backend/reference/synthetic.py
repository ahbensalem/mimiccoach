"""Synthetic reference library generator.

Also exposes `landmarks_for_entry(entry_id)` so the live `/analyze`
handler can ship the matched pro's pose JSON back to the frontend
overlay (we don't store landmarks in the Qdrant payload — they'd
balloon storage 100×). Reproduces clip[id] deterministically.

Produces a deterministic set of MediaPipe-33 pose sequences spanning all
seven motions, with parametric variation (timing, amplitude, body
proportions, handedness). The generator runs the same P1+P2 pipeline
that user uploads do, so the resulting phase_tokens come from the
identical encoding path — retrieval against a real user clip behaves
the same way it would against real reference data.

Why we ship a synthetic library:

  * The demo runs end-to-end on a fresh checkout with no external
    downloads, registrations, or licensing review. Real data is a
    quality upgrade layered on top, not a blocker for the demo.
  * Synthetic clips give us a controlled distribution over
    skill_level × body_type so the filter chips have something to
    actually filter against from day one.
  * The synthetic generators encode the same phase structure that
    motions.yaml declares — which doubles as a sanity check on the
    segmenter.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from pipeline.embed import HandCraftedEmbedder, phase_tokens
from pipeline.segment import segment_video
from pipeline.skeleton_map import MP_INDEX

DEFAULT_FPS: float = 30.0

# ---------------------------------------------------------------------------
# Body templates: vary shoulder/hip widths so body_type distributions are real.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BodyTemplate:
    name: str
    body_type: str
    shoulder_w: float
    hip_w: float
    height: float = 0.70  # ankle.y - nose.y in normalized image space


BODY_TEMPLATES: tuple[BodyTemplate, ...] = (
    BodyTemplate("narrow_lean",      "narrow",   shoulder_w=0.16, hip_w=0.16, height=0.74),
    BodyTemplate("narrow_compact",   "narrow",   shoulder_w=0.15, hip_w=0.16, height=0.66),
    BodyTemplate("balanced_classic", "balanced", shoulder_w=0.18, hip_w=0.16, height=0.72),
    BodyTemplate("balanced_tall",    "balanced", shoulder_w=0.19, hip_w=0.17, height=0.78),
    BodyTemplate("broad_strong",     "broad",    shoulder_w=0.24, hip_w=0.19, height=0.70),
    BodyTemplate("broad_compact",    "broad",    shoulder_w=0.22, hip_w=0.18, height=0.62),
)


def _base_pose(body: BodyTemplate) -> np.ndarray:
    """Static standing pose at body's proportions, in normalized image space."""
    base = np.zeros((33, 4), dtype=np.float32)
    base[:, 3] = 0.95  # visibility

    cx = 0.50
    nose_y = 0.16
    hip_y = nose_y + body.height * 0.55
    ankle_y = nose_y + body.height
    shoulder_y = nose_y + body.height * 0.18
    elbow_y = nose_y + body.height * 0.36
    wrist_y = nose_y + body.height * 0.52
    knee_y = nose_y + body.height * 0.78

    base[MP_INDEX["nose"]]            = [cx,                          nose_y,    0.0, 0.95]
    base[MP_INDEX["left_shoulder"]]   = [cx - body.shoulder_w / 2,    shoulder_y, 0.0, 0.95]
    base[MP_INDEX["right_shoulder"]]  = [cx + body.shoulder_w / 2,    shoulder_y, 0.0, 0.95]
    base[MP_INDEX["left_elbow"]]      = [cx - body.shoulder_w / 2 - 0.02, elbow_y, 0.0, 0.95]
    base[MP_INDEX["right_elbow"]]     = [cx + body.shoulder_w / 2 + 0.02, elbow_y, 0.0, 0.95]
    base[MP_INDEX["left_wrist"]]      = [cx - body.shoulder_w / 2 - 0.04, wrist_y, 0.0, 0.95]
    base[MP_INDEX["right_wrist"]]     = [cx + body.shoulder_w / 2 + 0.04, wrist_y, 0.0, 0.95]
    base[MP_INDEX["left_hip"]]        = [cx - body.hip_w / 2,         hip_y,     0.0, 0.95]
    base[MP_INDEX["right_hip"]]       = [cx + body.hip_w / 2,         hip_y,     0.0, 0.95]
    base[MP_INDEX["left_knee"]]       = [cx - body.hip_w / 2,         knee_y,    0.0, 0.95]
    base[MP_INDEX["right_knee"]]      = [cx + body.hip_w / 2,         knee_y,    0.0, 0.95]
    base[MP_INDEX["left_ankle"]]      = [cx - body.hip_w / 2,         ankle_y,   0.0, 0.95]
    base[MP_INDEX["right_ankle"]]     = [cx + body.hip_w / 2,         ankle_y,   0.0, 0.95]
    return base


def _arc(t: np.ndarray, start: int, end: int, amplitude: float) -> np.ndarray:
    """A half-sine bump from `start` to `end` of given peak amplitude."""
    out = np.zeros_like(t, dtype=np.float32)
    if end <= start:
        return out
    in_window = (t >= start) & (t <= end)
    phase = np.pi * (t - start) / max(1, end - start)
    out[in_window] = amplitude * np.sin(phase)[in_window]
    return out


# ---------------------------------------------------------------------------
# Per-motion generators
# ---------------------------------------------------------------------------

def _tennis_serve(body: BodyTemplate, T: int, *, skill: float, rng: np.random.Generator) -> np.ndarray:
    """skill ∈ [0, 1]: 0 = sloppy beginner, 1 = polished pro."""
    clip = np.tile(_base_pose(body), (T, 1, 1))
    t = np.arange(T)
    jitter = (1.0 - skill) * 0.02

    # Toss: left wrist rises early
    toss_amp = 0.18 + 0.04 * skill + rng.uniform(-0.02, 0.02)
    clip[:, MP_INDEX["left_wrist"], 1] += -_arc(t, int(T * 0.15), int(T * 0.45), toss_amp)

    # Trophy: right elbow lifts
    elbow_amp = 0.12 + 0.06 * skill
    clip[:, MP_INDEX["right_elbow"], 1] += -_arc(t, int(T * 0.40), int(T * 0.65), elbow_amp)

    # Contact: right wrist peaks
    contact_amp = 0.22 + 0.10 * skill
    clip[:, MP_INDEX["right_wrist"], 1] += -_arc(t, int(T * 0.55), int(T * 0.92), contact_amp)

    if jitter > 0:
        clip[..., :3] += rng.normal(0, jitter, size=clip[..., :3].shape).astype(np.float32)
    return clip


def _tennis_groundstroke(
    body: BodyTemplate, T: int, *, skill: float, backhand: bool, rng: np.random.Generator
) -> np.ndarray:
    clip = np.tile(_base_pose(body), (T, 1, 1))
    t = np.arange(T)
    jitter = (1.0 - skill) * 0.02

    # Take-back: hitting wrist swings back (negative x for forehand on right side)
    sign = +1.0 if backhand else -1.0
    swing_amp_x = 0.18 + 0.06 * skill + rng.uniform(-0.02, 0.02)
    clip[:, MP_INDEX["right_wrist"], 0] += sign * _arc(t, int(T * 0.10), int(T * 0.45), swing_amp_x)
    # Forward swing: wrist accelerates the other way through contact
    clip[:, MP_INDEX["right_wrist"], 0] += -sign * _arc(t, int(T * 0.45), int(T * 0.85), swing_amp_x * 1.1)
    # Slight upward sweep for follow-through
    clip[:, MP_INDEX["right_wrist"], 1] += -_arc(t, int(T * 0.55), int(T * 0.95), 0.10 + 0.05 * skill)

    if jitter > 0:
        clip[..., :3] += rng.normal(0, jitter, size=clip[..., :3].shape).astype(np.float32)
    return clip


def _vertical_motion(
    body: BodyTemplate, T: int, *, skill: float, hip_amp: float, wrist_amp: float, rng: np.random.Generator
) -> np.ndarray:
    """Squat-like motion: hips descend then ascend, optionally arms move with."""
    clip = np.tile(_base_pose(body), (T, 1, 1))
    t = np.arange(T)
    jitter = (1.0 - skill) * 0.02

    # Hips descend then ascend
    descent = _arc(t, int(T * 0.20), int(T * 0.55), hip_amp + 0.05 * skill)
    ascent = _arc(t, int(T * 0.55), int(T * 0.85), -(hip_amp + 0.05 * skill))
    bump = descent + ascent
    for j in ("left_hip", "right_hip", "left_knee", "right_knee"):
        clip[:, MP_INDEX[j], 1] += bump

    # Wrists track the hips for bench/row variants
    if abs(wrist_amp) > 1e-6:
        wrist_bump = _arc(t, int(T * 0.20), int(T * 0.55), wrist_amp) + _arc(
            t, int(T * 0.55), int(T * 0.85), -wrist_amp
        )
        for j in ("left_wrist", "right_wrist"):
            clip[:, MP_INDEX[j], 1] += wrist_bump

    if jitter > 0:
        clip[..., :3] += rng.normal(0, jitter, size=clip[..., :3].shape).astype(np.float32)
    return clip


def _golf_swing(body: BodyTemplate, T: int, *, skill: float, rng: np.random.Generator) -> np.ndarray:
    clip = np.tile(_base_pose(body), (T, 1, 1))
    t = np.arange(T)
    jitter = (1.0 - skill) * 0.02

    # Backswing: hands rise (above shoulders at top)
    rise_amp = 0.30 + 0.08 * skill
    clip[:, MP_INDEX["right_wrist"], 1] += -_arc(t, int(T * 0.10), int(T * 0.40), rise_amp)
    clip[:, MP_INDEX["left_wrist"], 1] += -_arc(t, int(T * 0.10), int(T * 0.40), rise_amp)

    # Downswing: hands accelerate horizontally (impact x peak around 60% of T)
    impact_amp = 0.22 + 0.10 * skill
    clip[:, MP_INDEX["right_wrist"], 0] += _arc(t, int(T * 0.40), int(T * 0.68), impact_amp)
    clip[:, MP_INDEX["left_wrist"], 0] += _arc(t, int(T * 0.40), int(T * 0.68), impact_amp)

    # Finish: hands rise on opposite side
    finish_amp = 0.18 + 0.05 * skill
    clip[:, MP_INDEX["right_wrist"], 1] += -_arc(t, int(T * 0.65), int(T * 0.95), finish_amp)

    if jitter > 0:
        clip[..., :3] += rng.normal(0, jitter, size=clip[..., :3].shape).astype(np.float32)
    return clip


# ---------------------------------------------------------------------------
# Per-motion catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SyntheticMotionPlan:
    motion: str
    sport: str
    durations: tuple[int, ...]
    """Frame counts per generated variation."""

    def make(
        self, body: BodyTemplate, T: int, *, skill: float, rng: np.random.Generator
    ) -> np.ndarray:
        if self.motion == "tennis_serve":
            return _tennis_serve(body, T, skill=skill, rng=rng)
        if self.motion == "tennis_forehand":
            return _tennis_groundstroke(body, T, skill=skill, backhand=False, rng=rng)
        if self.motion == "tennis_backhand":
            return _tennis_groundstroke(body, T, skill=skill, backhand=True, rng=rng)
        if self.motion == "fitness_squat":
            return _vertical_motion(body, T, skill=skill, hip_amp=0.10, wrist_amp=0.0, rng=rng)
        if self.motion == "fitness_bench_press":
            return _vertical_motion(body, T, skill=skill, hip_amp=0.0, wrist_amp=0.18, rng=rng)
        if self.motion == "fitness_bent_over_row":
            return _vertical_motion(body, T, skill=skill, hip_amp=0.0, wrist_amp=-0.12, rng=rng)
        if self.motion == "golf_swing":
            return _golf_swing(body, T, skill=skill, rng=rng)
        raise ValueError(f"unknown motion: {self.motion!r}")


PLANS: tuple[SyntheticMotionPlan, ...] = (
    SyntheticMotionPlan("tennis_serve",         "tennis",  durations=(75, 90, 105, 90, 100, 80)),
    SyntheticMotionPlan("tennis_forehand",      "tennis",  durations=(60, 70, 80, 65, 75)),
    SyntheticMotionPlan("tennis_backhand",      "tennis",  durations=(60, 70, 75, 65, 80)),
    SyntheticMotionPlan("fitness_squat",        "fitness", durations=(70, 90, 80, 95, 85)),
    SyntheticMotionPlan("fitness_bench_press",  "fitness", durations=(60, 75, 70, 80, 65)),
    SyntheticMotionPlan("fitness_bent_over_row","fitness", durations=(60, 70, 65, 80, 75)),
    SyntheticMotionPlan("golf_swing",           "golf",    durations=(80, 100, 90, 110, 95)),
)


@dataclass
class SyntheticEntry:
    id: int
    phase_tokens: list[list[float]]
    payload: dict[str, Any]


def _skill_label(skill: float) -> str:
    if skill < 0.40:
        return "beginner"
    if skill < 0.75:
        return "intermediate"
    return "pro"


SYNTHETIC_ATHLETE_NAMES: dict[str, list[str]] = {
    "pro": [
        "Aurora Vance", "Marcus Briggs", "Selena Park",
        "Theo Halvorsen", "Ines Castello", "Jules Whitlow",
    ],
    "intermediate": [
        "Cam Raja", "Dani Boatright", "Henry Quill",
        "Lucia Sandel", "Owen Tate", "Rey Estabrook",
    ],
    "beginner": [
        "Sam Doyle", "Mira Voss", "Kit Lestrange",
        "Pip Ardmore", "Tomi Reign", "Vesper Cole",
    ],
}


@lru_cache(maxsize=1)
def _all_landmarks(seed: int = 1729) -> dict[int, tuple[np.ndarray, str]]:
    """Cache every synthetic clip's landmarks by id, computed once.

    Identical iteration order + rng state to `generate()`, so the cached
    landmarks are bit-identical to what produced the embeddings stored
    in Qdrant. Used by `landmarks_for_entry()` to ship the matched pro's
    pose JSON back through `/analyze`.
    """
    rng = np.random.default_rng(seed)
    out: dict[int, tuple[np.ndarray, str]] = {}
    next_id = 1
    for plan in PLANS:
        for body in BODY_TEMPLATES:
            for i, T in enumerate(plan.durations):
                skill = (0.30, 0.55, 0.80, 0.45, 0.85, 0.65)[i % 6]
                clip = plan.make(body, T, skill=skill, rng=rng)
                out[next_id] = (clip, plan.motion)
                next_id += 1
    return out


def landmarks_for_entry(entry_id: int) -> tuple[np.ndarray, str, float] | None:
    """Recreate landmarks + motion + fps for a synthetic library id.

    Returns None for unknown / non-synthetic ids.
    """
    info = _all_landmarks().get(entry_id)
    if info is None:
        return None
    clip, motion = info
    return clip, motion, DEFAULT_FPS


def generate(seed: int = 1729) -> Iterator[SyntheticEntry]:
    """Yield synthetic SyntheticEntry rows covering all 7 motions.

    The output is deterministic for a given seed — same id, same phase
    tokens, same payload across runs.
    """
    embedder = HandCraftedEmbedder()
    landmarks_by_id = _all_landmarks(seed)
    next_id = 1
    for plan in PLANS:
        for body in BODY_TEMPLATES:
            for i, T in enumerate(plan.durations):
                skill = (0.30, 0.55, 0.80, 0.45, 0.85, 0.65)[i % 6]
                clip, motion_key = landmarks_by_id[next_id]
                assert motion_key == plan.motion, "landmark cache desync"
                _ = T  # T is encoded in clip.shape[0]; loop var kept for clarity

                boundaries = segment_video(clip, motion=plan.motion)
                per_frame = embedder.embed_frames(clip)
                tokens = phase_tokens(per_frame, boundaries)
                token_array = np.stack([t[1] for t in tokens], axis=0)

                skill_lbl = _skill_label(skill)
                athlete = SYNTHETIC_ATHLETE_NAMES[skill_lbl][
                    next_id % len(SYNTHETIC_ATHLETE_NAMES[skill_lbl])
                ]
                yield SyntheticEntry(
                    id=next_id,
                    phase_tokens=token_array.tolist(),
                    payload={
                        "sport": plan.sport,
                        "motion": plan.motion,
                        "skill_level": skill_lbl,
                        "body_type": body.body_type,
                        "athlete": athlete,
                        "source": "synthetic",
                        "source_url": None,
                        "license_note": "synthetic-bootstrap-data",
                    },
                )
                next_id += 1
