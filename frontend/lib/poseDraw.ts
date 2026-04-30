// Canvas drawing helpers for MediaPipe-33 skeletons.
//
// Landmarks come back from MediaPipe in normalized image-space coordinates
// (x, y in [0, 1] relative to the source video frame). The on-page <video>
// uses `object-contain`, which letterboxes the video inside its container —
// so we project landmarks into the *actual displayed video rect*, not the
// full canvas. Without this correction the skeleton drifts into the
// letterbox bands and "doesn't follow the person".

export type Pose = number[][]; // shape (33, 4) — [x, y, z, visibility]

export interface DrawRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

// MediaPipe Pose connections (subset that matters for sport motions).
const CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  // Head outline
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  // Torso
  [11, 12],
  [11, 23], [12, 24],
  [23, 24],
  // Left arm
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  // Right arm
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  // Left leg
  [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
  // Right leg
  [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
];

const PHASE_COLORS = [
  "#dc2626", // accent red — start
  "#f97316", // orange
  "#eab308", // amber
  "#16a34a", // green
  "#2563eb", // blue
  "#7c3aed", // violet
];

export function colorForPhase(phaseIndex: number, totalPhases: number): string {
  if (totalPhases <= 0) return PHASE_COLORS[0];
  const t = totalPhases > 1 ? phaseIndex / (totalPhases - 1) : 0;
  const idx = Math.min(
    PHASE_COLORS.length - 1,
    Math.floor(t * (PHASE_COLORS.length - 1) + 0.5),
  );
  return PHASE_COLORS[idx];
}

/**
 * Compute the on-screen rect of an `object-contain` <video> inside an
 * `outerW × outerH` container. Returns the full container rect when the
 * video has no intrinsic dimensions yet (still loading).
 */
export function getVideoDisplayRect(
  video: HTMLVideoElement | null,
  outerW: number,
  outerH: number,
): DrawRect {
  const naturalW = video?.videoWidth ?? 0;
  const naturalH = video?.videoHeight ?? 0;
  if (!naturalW || !naturalH) return { x: 0, y: 0, w: outerW, h: outerH };
  const naturalAspect = naturalW / naturalH;
  const outerAspect = outerW / outerH;
  if (naturalAspect > outerAspect) {
    // Video is relatively wider — fits to width, letterboxed top/bottom.
    const w = outerW;
    const h = w / naturalAspect;
    return { x: 0, y: (outerH - h) / 2, w, h };
  }
  // Video is relatively taller — fits to height, letterboxed left/right.
  const h = outerH;
  const w = h * naturalAspect;
  return { x: (outerW - w) / 2, y: 0, w, h };
}

/**
 * Compute the rect for a "skeleton-only" panel (no video). Centers a 9:16
 * box so the synthetic pro skeleton renders in a sensible portion of the
 * panel rather than across the whole container.
 */
export function getSkeletonOnlyRect(
  outerW: number,
  outerH: number,
  ratio = 9 / 16,
): DrawRect {
  const outerAspect = outerW / outerH;
  if (ratio > outerAspect) {
    const w = outerW;
    const h = w / ratio;
    return { x: 0, y: (outerH - h) / 2, w, h };
  }
  const h = outerH;
  const w = h * ratio;
  return { x: (outerW - w) / 2, y: 0, w, h };
}

export function drawPose(
  ctx: CanvasRenderingContext2D,
  pose: Pose,
  rect: DrawRect,
  options: { color?: string; minVisibility?: number; lineWidth?: number; jointRadius?: number } = {},
): void {
  const { x, y, w, h } = rect;
  const color = options.color ?? "#dc2626";
  const minVis = options.minVisibility ?? 0.4;
  const lineWidth = options.lineWidth ?? 3;
  const jointRadius = options.jointRadius ?? 4;

  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  // Bones
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  for (const [a, b] of CONNECTIONS) {
    const la = pose[a];
    const lb = pose[b];
    if (!la || !lb) continue;
    if ((la[3] ?? 1) < minVis || (lb[3] ?? 1) < minVis) continue;
    ctx.beginPath();
    ctx.moveTo(x + la[0] * w, y + la[1] * h);
    ctx.lineTo(x + lb[0] * w, y + lb[1] * h);
    ctx.stroke();
  }

  // Joints
  ctx.fillStyle = color;
  for (const lm of pose) {
    if (!lm) continue;
    if ((lm[3] ?? 1) < minVis) continue;
    ctx.beginPath();
    ctx.arc(x + lm[0] * w, y + lm[1] * h, jointRadius, 0, Math.PI * 2);
    ctx.fill();
  }
}
