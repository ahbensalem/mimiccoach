"use client";

import { useEffect, useMemo, useState } from "react";

import { CoachingTip } from "./CoachingTip";
import { PhaseScores } from "./PhaseScores";
import { SplitVideo } from "./SplitVideo";
import type { AnalyzeResponse, PhaseWindow } from "@/lib/types";

function equalPartitionPhases(totalFrames: number, n: number): PhaseWindow[] {
  if (n <= 0 || totalFrames <= 0) return [];
  const out: PhaseWindow[] = [];
  for (let i = 0; i < n; i++) {
    const start = Math.floor((i / n) * totalFrames);
    const end = Math.floor(((i + 1) / n) * totalFrames);
    out.push({ name: `p${i + 1}`, start_frame: start, end_frame: Math.max(end, start + 1) });
  }
  return out;
}

export function AnalysisResult({
  result,
  userVideo,
  onReset,
}: {
  result: AnalyzeResponse;
  userVideo: File;
  onReset: () => void;
}) {
  const [userVideoUrl, setUserVideoUrl] = useState<string | null>(null);

  useEffect(() => {
    const url = URL.createObjectURL(userVideo);
    setUserVideoUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [userVideo]);

  const userFps = result.user.fps ?? 30;
  const proFps = result.match?.fps ?? userFps;
  const userPhases = result.user.phases ?? [];
  // For the pro side we don't have segmented phase windows from the backend
  // yet, but the pro pose array runs at proFps; equal-partition its frame
  // count into the same number of phases as the user so the color coding
  // advances at proportional progress on both sides.
  const proPoses = result.match?.pose ?? null;
  const proPhases = proPoses
    ? equalPartitionPhases(proPoses.length, userPhases.length || 5)
    : userPhases;

  const score = result.match?.score ?? 0;
  const detected = result.user.detected_frames;
  const total = result.user.num_frames;
  const detectionRate = useMemo(
    () => (detected != null && total ? detected / total : null),
    [detected, total],
  );

  return (
    <section className="mx-auto w-full max-w-5xl space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-2xl font-semibold">
            {result.match?.athlete ?? "No match yet"}
          </h2>
          <p className="text-sm text-ink/60">
            {result.match
              ? <>Aggregate MaxSim <span className="font-medium tabular-nums text-ink">{score.toFixed(3)}</span> across {result.per_phase_scores.length} phases · {result.user.motion.replace(/_/g, " ")}</>
              : (result.error ?? "No reference clip matched the filters.")}
          </p>
        </div>
        <button
          type="button"
          onClick={onReset}
          className="rounded-full border border-ink/15 bg-white px-4 py-2 text-sm hover:bg-ink/5"
        >
          ← Try another clip
        </button>
      </header>

      <SplitVideo
        user={{
          videoUrl: userVideoUrl,
          poses: result.user.pose ?? null,
          phases: userPhases,
          fps: userFps,
          label: "Your motion",
          caption: detectionRate != null
            ? `${(detectionRate * 100).toFixed(0)}% pose detection across ${total} frames`
            : undefined,
        }}
        pro={{
          videoUrl: result.match?.video_url ?? null,
          poses: proPoses,
          phases: proPhases,
          fps: proFps,
          label: result.match?.athlete ? `Pro · ${result.match.athlete}` : "Pro",
          caption: result.match?.skill_level
            ? `${result.match.skill_level} reference`
            : undefined,
        }}
      />

      <PhaseScores
        scores={result.per_phase_scores}
        weakestPhase={result.weakest_phase}
      />

      <CoachingTip
        weakestPhase={result.weakest_phase}
        tip={result.coaching_tip}
        matchAthlete={result.match?.athlete ?? null}
      />

      <div className="rounded-3xl border border-ink/10 bg-canvas p-4 text-xs text-ink/60">
        <div className="font-medium text-ink/80">Filters applied</div>
        <div className="mt-1 flex flex-wrap gap-2">
          <span className="rounded-full border border-ink/10 bg-white px-2 py-0.5">sport: {result.filters_applied.sport}</span>
          <span className="rounded-full border border-ink/10 bg-white px-2 py-0.5">motion: {result.filters_applied.motion}</span>
          <span className="rounded-full border border-ink/10 bg-white px-2 py-0.5">skill_level: {result.filters_applied.skill_level ?? "any"}</span>
          <span className="rounded-full border border-ink/10 bg-white px-2 py-0.5">body_type: {result.filters_applied.body_type ?? "any"}</span>
          <span className="rounded-full border border-ink/10 bg-white px-2 py-0.5">detected as: {result.user.body_type}</span>
        </div>
      </div>
    </section>
  );
}
