"use client";

export function CoachingTip({
  weakestPhase,
  tip,
  matchAthlete,
}: {
  weakestPhase: string;
  tip: string;
  matchAthlete: string | null;
}) {
  return (
    <div className="rounded-3xl border border-ink/10 bg-white p-5">
      <div className="flex items-center gap-2 text-xs uppercase tracking-widest text-ink/50">
        <span className="rounded-full bg-amber/15 px-2 py-0.5 font-semibold text-amber">
          Focus phase
        </span>
        <span>{prettyPhase(weakestPhase)}</span>
      </div>
      <p className="mt-3 text-base leading-relaxed text-ink">{tip}</p>
      {matchAthlete ? (
        <p className="mt-3 text-xs text-ink/50">
          Comparison drawn against <span className="font-medium text-ink/70">{matchAthlete}</span>.
        </p>
      ) : null}
    </div>
  );
}

function prettyPhase(name: string): string {
  return name
    .split("_")
    .map((p) => p[0]?.toUpperCase() + p.slice(1))
    .join(" ");
}
