// Stub — full pose-drawing helpers land in P6.
// Will: draw MediaPipe-33 skeleton on a canvas, color edges by current phase,
// support pixel-space (from clip metadata) and normalized-space inputs.

export type Landmark = { x: number; y: number; z?: number; visibility?: number };
export type Pose = Landmark[];
