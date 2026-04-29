// Stub — passthrough to Modal so the browser doesn't hold the backend URL directly.
// Real implementation lands in P5/P6 (multipart upload forwarding, signed URL rewrite).

import { NextResponse } from "next/server";

export async function GET() {
  const backend = process.env.MODAL_BACKEND_URL;
  if (!backend) {
    return NextResponse.json(
      { error: "MODAL_BACKEND_URL not configured" },
      { status: 503 },
    );
  }
  try {
    const res = await fetch(`${backend}/healthz`, { cache: "no-store" });
    const body = await res.json();
    return NextResponse.json(body, { status: res.status });
  } catch (err) {
    return NextResponse.json(
      { error: "backend unreachable", detail: String(err) },
      { status: 502 },
    );
  }
}
