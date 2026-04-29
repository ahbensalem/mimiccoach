"use client";

import clsx from "clsx";

import type { PhaseScore } from "@/lib/types";

export function PhaseScores({
  scores,
  weakestPhase,
}: {
  scores: PhaseScore[];
  weakestPhase: string;
}) {
  if (!scores.length) return null;
  return (
    <div className="space-y-2">
      <div className="text-xs uppercase tracking-widest text-ink/50">
        Per-phase MaxSim · Qdrant late-interaction
      </div>
      <div className="grid grid-cols-2 gap-2 md:grid-cols-3 lg:grid-cols-5">
        {scores.map((s) => {
          const isWeakest = s.phase === weakestPhase;
          return (
            <div
              key={s.phase}
              className={clsx(
                "relative rounded-2xl border p-3 text-sm",
                isWeakest
                  ? "border-amber bg-amber/10"
                  : "border-ink/10 bg-white",
              )}
            >
              <div className="flex items-center justify-between text-xs uppercase tracking-widest text-ink/50">
                <span>{prettyPhase(s.phase)}</span>
                {isWeakest ? (
                  <span className="font-semibold text-amber">weakest</span>
                ) : null}
              </div>
              <div className="mt-1 text-xl font-semibold tabular-nums">
                {s.score.toFixed(2)}
              </div>
              <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-ink/10">
                <div
                  className={clsx(
                    "h-full rounded-full transition-[width]",
                    isWeakest ? "bg-amber" : "bg-ink",
                  )}
                  style={{ width: `${Math.max(0, Math.min(100, s.score * 100))}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function prettyPhase(name: string): string {
  return name
    .split("_")
    .map((p) => p[0]?.toUpperCase() + p.slice(1))
    .join(" ");
}
