// POST /api/proxy/analyze — forward a multipart upload to the Modal backend.
// The browser uploads here; we re-stream the multipart body to Modal so
// MODAL_BACKEND_URL never lands in client code.

import { NextResponse } from "next/server";

import { backendOrError } from "@/lib/proxy";

// Defaults to 1 MB on Vercel; phone clips can be 20-30 MB.
export const config = { api: { bodyParser: false } };
export const runtime = "nodejs";
export const maxDuration = 60;

export async function POST(req: Request) {
  const backend = backendOrError();
  if (backend.error) return backend.error;

  // FormData round-trips multipart cleanly; we re-encode rather than streaming
  // the raw body so the boundary is regenerated for the upstream request.
  const incoming = await req.formData();
  const upstream = new FormData();
  for (const [key, value] of incoming.entries()) {
    upstream.append(key, value);
  }

  try {
    const res = await fetch(`${backend.url}/analyze`, {
      method: "POST",
      body: upstream,
    });
    if (!res.ok) {
      return NextResponse.json(
        { error: "analyze failed", status: res.status, detail: await res.text() },
        { status: res.status },
      );
    }
    return NextResponse.json(await res.json(), { status: 200 });
  } catch (err) {
    return NextResponse.json(
      { error: "backend unreachable", detail: String(err) },
      { status: 502 },
    );
  }
}
