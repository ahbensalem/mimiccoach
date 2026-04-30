// Shared types between the API proxy and the React components.
// Mirrors the JSON shape produced by analyze_from_landmarks() in
// backend/app.py.

export type Sport = "tennis" | "fitness" | "golf";
export type SkillLevel = "beginner" | "intermediate" | "pro";
export type BodyType = "narrow" | "balanced" | "broad";

export interface MotionSummary {
  key: string;            // e.g. "tennis_serve"
  sport: Sport;
  label: string;          // human-readable
  phases: string[];
  is_hero: boolean;
}

export interface MotionsResponse {
  motions: MotionSummary[];
}

export interface PhaseWindow {
  name: string;
  start_frame: number;
  end_frame: number;
}

export interface UserMeta {
  sport: Sport;
  motion: string;
  body_type: BodyType;
  phases: PhaseWindow[];
  fps?: number;
  num_frames?: number;
  detected_frames?: number;
  width?: number;
  height?: number;
  pose?: number[][][];   // (T, 33, 4) inline landmarks
}

export interface MatchInfo {
  point_id: string;
  score: number;
  athlete: string | null;
  source_url: string | null;
  skill_level: SkillLevel | null;
  body_type: BodyType | null;
  pose_url: string | null;
  video_url: string | null;
  /** Inline pose JSON for the matched pro clip — present for synthetic
   *  library matches and (eventually) for real reference clips that ship
   *  a pose blob. (T, 33, 4) per-frame landmarks. */
  pose: number[][][] | null;
  /** fps of the inlined `pose` array. */
  fps: number | null;
}

export interface PhaseScore {
  phase: string;
  score: number;
}

export interface FiltersApplied {
  sport: Sport;
  motion: string;
  skill_level: SkillLevel | null;
  body_type: BodyType | null;
}

export interface AlternativeMatch {
  point_id: string;
  score: number;
  athlete: string | null;
  skill_level: SkillLevel | null;
}

export interface AnalyzeResponse {
  user: UserMeta;
  filters_applied: FiltersApplied;
  match: MatchInfo | null;
  per_phase_scores: PhaseScore[];
  weakest_phase: string;
  coaching_tip: string;
  alternatives: AlternativeMatch[];
  error?: string;
}

// Static fallback motions list — matches backend/pipeline/motions.yaml.
// Used when the backend is unreachable so the upload form still works.
export const FALLBACK_MOTIONS: MotionSummary[] = [
  { key: "tennis_serve",         sport: "tennis",  label: "Serve",         phases: ["stance", "toss", "trophy", "contact", "follow_through"], is_hero: true  },
  { key: "tennis_forehand",      sport: "tennis",  label: "Forehand",      phases: ["ready", "take_back", "forward_swing", "contact", "follow_through"], is_hero: false },
  { key: "tennis_backhand",      sport: "tennis",  label: "Backhand",      phases: ["ready", "take_back", "forward_swing", "contact", "follow_through"], is_hero: false },
  { key: "fitness_squat",        sport: "fitness", label: "Squat",         phases: ["setup", "descent", "bottom", "ascent", "lockout"], is_hero: false },
  { key: "fitness_bench_press",  sport: "fitness", label: "Bench press",   phases: ["unrack", "descent", "touch", "ascent", "lockout"], is_hero: false },
  { key: "fitness_bent_over_row",sport: "fitness", label: "Bent-over row", phases: ["hinge", "pull", "contraction", "eccentric", "reset"], is_hero: false },
  { key: "golf_swing",           sport: "golf",    label: "Full swing",    phases: ["address", "backswing", "top", "downswing", "impact", "finish"], is_hero: false },
];
