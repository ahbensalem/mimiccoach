"use client";

import { useEffect, useRef } from "react";

import {
  colorForPhase,
  drawPose,
  getSkeletonOnlyRect,
  getVideoDisplayRect,
  type Pose,
} from "@/lib/poseDraw";
import type { PhaseWindow } from "@/lib/types";

interface Props {
  poses: Pose[];          // (T, 33, 4) per-frame landmarks
  fps: number;            // frames-per-second of the pose array
  phases: PhaseWindow[];  // contiguous, sorted by start_frame
  containerRef: React.RefObject<HTMLElement | null>;
  /** Optional video reference. When provided, skeleton aligns to the
   *  letterboxed video rect inside the container. */
  videoRef?: React.RefObject<HTMLVideoElement | null>;
  /** Returns current time in seconds (for the *pose array's* clock).
   *  When omitted, uses videoRef.current.currentTime. */
  currentTimeFn?: () => number;
  /** Visual style — defaults are tuned for video-overlay (subtle); pass
   *  larger values for skeleton-only panels (e.g. the synthetic pro side). */
  lineWidth?: number;
  jointRadius?: number;
  alpha?: number;
  className?: string;
}

/**
 * Draws a MediaPipe-33 skeleton on a `<canvas>` overlaid on a `<video>` (or
 * a skeleton-only panel when no video is available). Color tracks which
 * phase the current frame falls inside.
 */
export function SkeletonCanvas({
  poses,
  fps,
  phases,
  containerRef,
  videoRef,
  currentTimeFn,
  lineWidth,
  jointRadius,
  alpha,
  className,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !poses.length) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    function resize() {
      if (!canvas || !container) return;
      const rect = container.getBoundingClientRect();
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const cssW = Math.max(1, Math.round(rect.width));
      const cssH = Math.max(1, Math.round(rect.height));
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
      canvas.style.width = `${cssW}px`;
      canvas.style.height = `${cssH}px`;
      ctx?.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function frameToPhaseIndex(frame: number): number {
      for (let i = 0; i < phases.length; i++) {
        if (frame < phases[i].end_frame) return i;
      }
      return phases.length - 1;
    }

    function loop() {
      if (!canvas || !container || !ctx) return;
      resize();

      const cssW = parseInt(canvas.style.width, 10);
      const cssH = parseInt(canvas.style.height, 10);
      ctx.clearRect(0, 0, cssW, cssH);

      let t: number;
      if (currentTimeFn) {
        t = currentTimeFn();
      } else {
        t = videoRef?.current?.currentTime ?? 0;
      }
      const frameIdx = Math.min(
        poses.length - 1,
        Math.max(0, Math.floor(t * fps)),
      );
      const pose = poses[frameIdx];
      if (pose) {
        const rect = videoRef
          ? getVideoDisplayRect(videoRef.current, cssW, cssH)
          : getSkeletonOnlyRect(cssW, cssH);
        const pi = phases.length ? frameToPhaseIndex(frameIdx) : 0;
        const color = phases.length ? colorForPhase(pi, phases.length) : "#dc2626";
        drawPose(ctx, pose, rect, {
          color,
          lineWidth: lineWidth ?? 2,
          jointRadius: jointRadius ?? 2.5,
          alpha: alpha ?? 0.9,
        });
      }

      rafRef.current = requestAnimationFrame(loop);
    }

    rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [poses, fps, phases, containerRef, videoRef, currentTimeFn, lineWidth, jointRadius, alpha]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
    />
  );
}
