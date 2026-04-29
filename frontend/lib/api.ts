// Client-side API helpers — call /api/proxy/* from React components.

import type { AnalyzeResponse, MotionsResponse, SkillLevel, BodyType } from "./types";

const API_BASE = "/api/proxy";

export async function fetchMotions(): Promise<MotionsResponse> {
  const res = await fetch(`${API_BASE}/motions`, { cache: "no-store" });
  if (!res.ok) throw new Error(`motions fetch failed: ${res.status}`);
  return res.json();
}

export interface AnalyzeArgs {
  video: File;
  motion: string;
  skill_level?: SkillLevel;
  body_type?: BodyType;
  limit?: number;
}

export async function analyzeVideo(args: AnalyzeArgs): Promise<AnalyzeResponse> {
  const fd = new FormData();
  fd.append("video", args.video);
  fd.append("motion", args.motion);
  if (args.skill_level) fd.append("skill_level", args.skill_level);
  if (args.body_type) fd.append("body_type", args.body_type);
  if (args.limit) fd.append("limit", String(args.limit));

  const res = await fetch(`${API_BASE}/analyze`, { method: "POST", body: fd });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`analyze failed: ${res.status} ${detail}`);
  }
  return res.json();
}

export async function checkHealth(): Promise<{ status: string }> {
  const res = await fetch(API_BASE, { cache: "no-store" });
  if (!res.ok) throw new Error(`health check failed: ${res.status}`);
  return res.json();
}
