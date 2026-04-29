"""Rule-based coaching tip generation.

Given per-phase MaxSim scores between the user's clip and the matched pro
clip, identify the *weakest* phase and emit a templated coaching tip.
Per the plan, this is intentionally rule-based (no LLM) — it's
deterministic on demo day, side-steps the "no chatbot" rule, and is
explainable to the judges.

The current implementation produces phase-aware generic tips. Once
real reference data and angle-delta computation lands (P7+), we can
upgrade to body-part-specific tips driven by joint-angle deltas at the
weakest phase ("your trophy elbow drops 18° earlier than the pro's").
"""
from __future__ import annotations

from dataclasses import dataclass

# Phase-scoped advice. Indexed by motion key, then phase name.
# Keep sentences short, action-oriented, and demo-friendly. The phrasing
# mimics what a coach would say in one sentence on a sideline.
_PHASE_TIPS: dict[str, dict[str, str]] = {
    "tennis_serve": {
        "stance": "Reset your stance — feet square to the baseline before you start the motion.",
        "toss": "Your toss arm rises late or off-line — release with a fully-extended arm at the top.",
        "trophy": "Trophy position needs more depth — coil the back, get the racket head higher behind you.",
        "contact": "Reach up at contact — drive through the ball with the racket head fully extended.",
        "follow_through": "Finish across the body — let the racket follow through past your opposite hip.",
    },
    "tennis_forehand": {
        "ready": "Get into a balanced ready position with knees bent and racket centered.",
        "take_back": "Earlier preparation — start the take-back the moment the ball leaves the opponent's racket.",
        "forward_swing": "Drive forward through the ball — don't stall the racket head into contact.",
        "contact": "Meet the ball out in front — your arm should be slightly extended at contact.",
        "follow_through": "Finish over the opposite shoulder — full kinetic chain, no aborted swings.",
    },
    "tennis_backhand": {
        "ready": "Stay neutral and balanced before you commit to the backhand side.",
        "take_back": "Turn your shoulders earlier — coil into the shot before the ball lands.",
        "forward_swing": "Brush up through the ball — don't stop the racket halfway through.",
        "contact": "Hit out in front, with the racket face square at contact.",
        "follow_through": "Full extension on the follow-through — finish high, across your dominant shoulder.",
    },
    "fitness_squat": {
        "setup": "Tighten your setup — feet planted, brace before you unrack.",
        "descent": "Descend slower and under control — drop your hips straight down, knees tracking over toes.",
        "bottom": "Hit depth with neutral spine — break parallel without losing your back angle.",
        "ascent": "Drive through your mid-foot — don't shift forward as you stand.",
        "lockout": "Stand fully tall and tight at lockout before you reset.",
    },
    "fitness_bench_press": {
        "unrack": "Set your shoulder blades retracted before you take the weight off the rack.",
        "descent": "Lower the bar with control — touch the chest at a consistent point.",
        "touch": "Brief pause at the chest — no bouncing the bar.",
        "ascent": "Press in a slight arc back over the shoulders, not straight up.",
        "lockout": "Lock out fully — elbows extended, scapulas still retracted.",
    },
    "fitness_bent_over_row": {
        "hinge": "Set the hinge — hips back, neutral spine, bar over mid-foot.",
        "pull": "Pull with the back, not the arms — drive the elbows back along the ribs.",
        "contraction": "Squeeze at the top before you release the rep.",
        "eccentric": "Control the descent — don't drop the bar back to start.",
        "reset": "Reset the hinge before each rep — don't rush the next pull.",
    },
    "golf_swing": {
        "address": "Set up balanced — weight even, club square behind the ball.",
        "backswing": "Take it back smoothly with a full shoulder turn, not just hands.",
        "top": "Stop the club at parallel or just short — overswinging leaks power.",
        "downswing": "Start the downswing from the ground up — hips, then torso, then arms.",
        "impact": "Compress the ball at impact — hands ahead of the clubhead, weight forward.",
        "finish": "Finish in balance, belt-buckle to the target.",
    },
}

_GENERIC_FALLBACK = (
    "Focus your next session on this phase — record again and compare."
)


@dataclass
class CoachingTip:
    weakest_phase: str
    score: float
    """The MaxSim score on the weakest phase (lower = more room to improve)."""
    tip: str


def coach_from_per_phase(
    motion: str,
    per_phase_scores: list[float],
    phase_names: list[str],
) -> CoachingTip:
    """Identify the weakest phase and emit a templated coaching tip.

    Args:
        motion: motion key (tennis_serve, fitness_squat, ...).
        per_phase_scores: list of MaxSim scores, one per phase, in clip order.
        phase_names: list of phase name strings in matching order.

    Returns:
        CoachingTip with weakest phase, its score, and a one-sentence tip.
    """
    if not per_phase_scores:
        raise ValueError("per_phase_scores must be non-empty")
    if len(per_phase_scores) != len(phase_names):
        raise ValueError(
            f"per_phase_scores length ({len(per_phase_scores)}) "
            f"!= phase_names length ({len(phase_names)})"
        )

    weakest_idx = min(range(len(per_phase_scores)), key=lambda i: per_phase_scores[i])
    weakest_phase = phase_names[weakest_idx]
    score = float(per_phase_scores[weakest_idx])

    tips = _PHASE_TIPS.get(motion, {})
    tip = tips.get(weakest_phase, _GENERIC_FALLBACK)

    return CoachingTip(weakest_phase=weakest_phase, score=score, tip=tip)
