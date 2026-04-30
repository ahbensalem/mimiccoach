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
 * Two synchronized panels (user + pro) with MediaPipe skeletons. The user
 * side has a real <video>; the pro side either has its own video URL (real
 * reference data) or animates a skeleton-only panel using the matched
 * synthetic pro's landmarks. Both skeletons are driven from the user
 * video's currentTime so the per-phase coloring on both sides advances
 * together — that's the demo's side-by-side moment.
 */
export function SplitVideo({ user, pro }: { user: Side; pro: Side }) {
  const userRef = useRef<HTMLVideoElement>(null);
  const proRef = useRef<HTMLVideoElement>(null);
  const userPanelRef = useRef<HTMLDivElement>(null);
  const proPanelRef = useRef<HTMLDivElement>(null);
  const [duration, setDuration] = useState(1);
  const [t, setT] = useState(0);

  // Sync the pro <video> (when present) to the user's progress fraction so
  // clips of different lengths still phase-line up.
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

  // Pro-side clock: maps the user's progress to a time inside the pro pose
  // array's own duration, so the pro skeleton animates in sync without a
  // pro video element.
  const proDuration = pro.poses ? pro.poses.length / Math.max(1, pro.fps) : 0;
  const userDurationRef = useRef(duration);
  userDurationRef.current = duration;
  const userTimeRef = useRef(t);
  userTimeRef.current = t;
  function proCurrentTime(): number {
    const ud = userDurationRef.current;
    const ut = userTimeRef.current;
    if (ud <= 0 || proDuration <= 0) return 0;
    return Math.min(proDuration - 0.001, (ut / ud) * proDuration);
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <div className="space-y-1">
          <div className="text-xs uppercase tracking-widest text-ink/50">{user.label}</div>
          <div
            ref={userPanelRef}
            className="relative aspect-[9/16] overflow-hidden rounded-2xl border border-ink/10 bg-ink md:aspect-video"
          >
            {user.videoUrl ? (
              <video
                ref={userRef}
                src={user.videoUrl}
                playsInline
                muted
                className="h-full w-full object-contain"
              />
            ) : null}
            {user.poses && user.phases.length ? (
              <SkeletonCanvas
                poses={user.poses}
                fps={user.fps}
                phases={user.phases}
                containerRef={userPanelRef}
                videoRef={userRef}
              />
            ) : null}
          </div>
          {user.caption ? <p className="text-xs text-ink/60">{user.caption}</p> : null}
        </div>

        <div className="space-y-1">
          <div className="text-xs uppercase tracking-widest text-ink/50">{pro.label}</div>
          <div
            ref={proPanelRef}
            className="relative aspect-[9/16] overflow-hidden rounded-2xl border border-ink/10 bg-gradient-to-br from-ink to-ink/80 md:aspect-video"
          >
            {pro.videoUrl ? (
              <video
                ref={proRef}
                src={pro.videoUrl}
                playsInline
                muted
                className="h-full w-full object-contain"
              />
            ) : pro.poses ? (
              <div className="absolute bottom-3 left-3 z-10 rounded-full bg-canvas/15 px-3 py-1 text-[11px] uppercase tracking-widest text-canvas/80 backdrop-blur">
                Synthetic reference · pose only
              </div>
            ) : (
              <div className="flex h-full w-full items-center justify-center px-4 text-center text-xs text-canvas/60">
                Pro reference clip lands once the library is built (P3).
              </div>
            )}
            {pro.poses && pro.phases.length ? (
              pro.videoUrl ? (
                <SkeletonCanvas
                  poses={pro.poses}
                  fps={pro.fps}
                  phases={pro.phases}
                  containerRef={proPanelRef}
                  videoRef={proRef}
                />
              ) : (
                <SkeletonCanvas
                  poses={pro.poses}
                  fps={pro.fps}
                  phases={pro.phases}
                  containerRef={proPanelRef}
                  currentTimeFn={proCurrentTime}
                />
              )
            ) : null}
          </div>
          {pro.caption ? <p className="text-xs text-ink/60">{pro.caption}</p> : null}
        </div>
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
