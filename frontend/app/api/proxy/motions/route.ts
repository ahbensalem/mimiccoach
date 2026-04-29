// GET /api/proxy/motions — list supported motions and phases.
// Cached for 60s on the Vercel edge so the upload form's motion picker
// renders instantly on cold loads.

import { NextResponse } from "next/server";

import { backendOrError } from "@/lib/proxy";

export const revalidate = 60;

export async function GET() {
  const backend = backendOrError();
  if (backend.error) return backend.error;

  try {
    const res = await fetch(`${backend.url}/motions`, {
      next: { revalidate: 60 },
    });
    const body = await res.json();
    return NextResponse.json(body, { status: res.status });
  } catch (err) {
    return NextResponse.json(
      { error: "backend unreachable", detail: String(err) },
      { status: 502 },
    );
  }
}
