// Server-side helper for the /api/proxy/* routes.
import { NextResponse } from "next/server";

export function backendOrError(): { url: string; error: null } | { url: null; error: NextResponse } {
  const url = process.env.MODAL_BACKEND_URL;
  if (!url) {
    return {
      url: null,
      error: NextResponse.json(
        { error: "MODAL_BACKEND_URL not configured" },
        { status: 503 },
      ),
    };
  }
  return { url, error: null };
}
