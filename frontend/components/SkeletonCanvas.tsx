"use client";

import { useEffect, useRef } from "react";

import { colorForPhase, drawPose, type Pose } from "@/lib/poseDraw";
import type { PhaseWindow } from "@/lib/types";

interface Props {
  poses: Pose[];          // (T, 33, 4) per-frame landmarks
  fps: number;
  phases: PhaseWindow[];  // contiguous, sorted by start_frame
  videoRef: React.RefObject<HTMLVideoElement | null>;
  /** Resize the canvas to the rendered video element each frame. */
  className?: string;
}

/**
 * Draws a MediaPipe-33 skeleton on a <canvas> overlaid on a <video>.
 * Color is driven by which phase the current frame falls inside.
 */
export function SkeletonCanvas({ poses, fps, phases, videoRef, className }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !poses.length) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    function resize() {
      if (!canvas || !video) return;
      const rect = video.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width));
      canvas.height = Math.max(1, Math.round(rect.height));
    }

    function frameToPhaseIndex(frame: number): number {
      for (let i = 0; i < phases.length; i++) {
        if (frame < phases[i].end_frame) return i;
      }
      return phases.length - 1;
    }

    function loop() {
      if (!canvas || !video) return;
      resize();
      const t = video.currentTime;
      const frameIdx = Math.min(poses.length - 1, Math.max(0, Math.floor(t * fps)));
      const pose = poses[frameIdx];
      if (pose) {
        const pi = phases.length ? frameToPhaseIndex(frameIdx) : 0;
        const color = phases.length ? colorForPhase(pi, phases.length) : "#dc2626";
        drawPose(ctx!, pose, canvas.width, canvas.height, { color });
      }
      rafRef.current = requestAnimationFrame(loop);
    }

    rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [poses, fps, phases, videoRef]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
    />
  );
}
