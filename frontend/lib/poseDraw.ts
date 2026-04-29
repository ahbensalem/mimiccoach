// Canvas drawing helpers for MediaPipe-33 skeletons.
//
// The MediaPipe Pose Landmarker returns 33 landmarks per frame in normalized
// image-space coordinates (x and y in [0, 1]). We project them to canvas
// pixels and draw a phase-colored skeleton on top of a <video> element.

export type Pose = number[][]; // shape (33, 4) — [x, y, z, visibility]

// MediaPipe Pose connections (subset that matters for sport motions).
// Each pair is (mediapipe_landmark_a, mediapipe_landmark_b).
const CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  // Head outline
  [0, 1], [1, 2], [2, 3], [3, 7],   // left eye + ear
  [0, 4], [4, 5], [5, 6], [6, 8],   // right eye + ear
  // Torso
  [11, 12],  // shoulders
  [11, 23], [12, 24],  // shoulder → hip
  [23, 24],  // hips
  // Left arm
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  // Right arm
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  // Left leg
  [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
  // Right leg
  [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
];

// Phase color palette. Keys are normalized phase positions (0 = first phase,
// 1 = last). The active phase determines the skeleton's color.
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
  // Map [0, totalPhases-1] → [0, PHASE_COLORS.length-1]
  const t = totalPhases > 1 ? phaseIndex / (totalPhases - 1) : 0;
  const idx = Math.min(
    PHASE_COLORS.length - 1,
    Math.floor(t * (PHASE_COLORS.length - 1) + 0.5),
  );
  return PHASE_COLORS[idx];
}

export function drawPose(
  ctx: CanvasRenderingContext2D,
  pose: Pose,
  width: number,
  height: number,
  options: { color?: string; minVisibility?: number; lineWidth?: number; jointRadius?: number } = {},
): void {
  const color = options.color ?? "#dc2626";
  const minVis = options.minVisibility ?? 0.4;
  const lineWidth = options.lineWidth ?? 3;
  const jointRadius = options.jointRadius ?? 4;

  ctx.clearRect(0, 0, width, height);
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
    ctx.moveTo(la[0] * width, la[1] * height);
    ctx.lineTo(lb[0] * width, lb[1] * height);
    ctx.stroke();
  }

  // Joints
  ctx.fillStyle = color;
  for (const lm of pose) {
    if (!lm) continue;
    if ((lm[3] ?? 1) < minVis) continue;
    ctx.beginPath();
    ctx.arc(lm[0] * width, lm[1] * height, jointRadius, 0, Math.PI * 2);
    ctx.fill();
  }
}
