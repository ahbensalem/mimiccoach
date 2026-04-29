"use client";

import { useEffect, useRef, useState } from "react";

import { SkeletonCanvas } from "./SkeletonCanvas";
import type { PhaseWindow } from "@/lib/types";

interface Side {
  videoUrl: string | null;
  poses: number[][][] | null;
  phases: PhaseWindow[];
  fps: number;
  label: string;
  caption?: string;
}

/**
 * Two synchronized <video> elements (user + pro) with MediaPipe skeletons
 * drawn on top via <SkeletonCanvas>. The user-side video drives the shared
 * scrubber; the pro side rate-aligns to the same normalized progress.
 *
 * If the pro side has no video URL, we render a skeleton-only animation
 * (driven by a hidden internal clock at the user's fps).
 */
export function SplitVideo({ user, pro }: { user: Side; pro: Side }) {
  const userRef = useRef<HTMLVideoElement>(null);
  const proRef = useRef<HTMLVideoElement>(null);
  const [duration, setDuration] = useState(1);
  const [t, setT] = useState(0);

  // Sync the pro video's currentTime to a fraction of its duration matching
  // the user's normalized progress, so phase-to-phase comparison lines up
  // even when the two clips have different lengths.
  useEffect(() => {
    const u = userRef.current;
    if (!u) return;
    const onTime = () => {
      setT(u.currentTime);
      const p = proRef.current;
      if (p && u.duration > 0 && p.duration > 0) {
        const progress = u.currentTime / u.duration;
        const target = Math.min(p.duration - 0.01, progress * p.duration);
        if (Math.abs(p.currentTime - target) > 0.05) p.currentTime = target;
      }
    };
    const onMeta = () => setDuration(Math.max(0.01, u.duration || 0.01));
    u.addEventListener("timeupdate", onTime);
    u.addEventListener("loadedmetadata", onMeta);
    return () => {
      u.removeEventListener("timeupdate", onTime);
      u.removeEventListener("loadedmetadata", onMeta);
    };
  }, []);

  // Mirror play/pause from user → pro.
  useEffect(() => {
    const u = userRef.current;
    const p = proRef.current;
    if (!u || !p) return;
    const onPlay = () => p.play().catch(() => {});
    const onPause = () => p.pause();
    u.addEventListener("play", onPlay);
    u.addEventListener("pause", onPause);
    return () => {
      u.removeEventListener("play", onPlay);
      u.removeEventListener("pause", onPause);
    };
  }, []);

  function scrubTo(value: number) {
    const u = userRef.current;
    if (u) u.currentTime = value;
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <SidePanel side={user} videoRef={userRef} />
        <SidePanel side={pro} videoRef={proRef} />
      </div>
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => {
            const u = userRef.current;
            if (!u) return;
            if (u.paused) u.play().catch(() => {});
            else u.pause();
          }}
          className="rounded-full bg-ink px-4 py-1.5 text-sm font-medium text-canvas"
        >
          ▶ / ❚❚
        </button>
        <input
          type="range"
          min={0}
          max={duration}
          step={0.01}
          value={t}
          onChange={(e) => scrubTo(parseFloat(e.target.value))}
          className="flex-1 accent-accent"
        />
        <span className="w-16 text-right text-xs tabular-nums text-ink/60">
          {t.toFixed(2)}s
        </span>
      </div>
    </div>
  );
}

function SidePanel({
  side,
  videoRef,
}: {
  side: Side;
  videoRef: React.RefObject<HTMLVideoElement | null>;
}) {
  return (
    <div className="space-y-1">
      <div className="text-xs uppercase tracking-widest text-ink/50">{side.label}</div>
      <div className="relative aspect-[9/16] overflow-hidden rounded-2xl border border-ink/10 bg-ink md:aspect-video">
        {side.videoUrl ? (
          <video
            ref={videoRef}
            src={side.videoUrl}
            playsInline
            muted
            className="h-full w-full object-contain"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center px-4 text-center text-xs text-canvas/60">
            Pro reference clip lands once the library is built (P3).
            <br />
            The retrieval and per-phase scoring are live below.
          </div>
        )}
        {side.poses && side.phases.length && side.videoUrl ? (
          <SkeletonCanvas
            poses={side.poses}
            fps={side.fps}
            phases={side.phases}
            videoRef={videoRef}
          />
        ) : null}
      </div>
      {side.caption ? (
        <p className="text-xs text-ink/60">{side.caption}</p>
      ) : null}
    </div>
  );
}
