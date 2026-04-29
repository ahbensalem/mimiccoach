// Stub — full client lands in P5/P6.
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "/api/proxy";

export async function getHealth(): Promise<{ status: string; service: string }> {
  const res = await fetch(API_BASE, { cache: "no-store" });
  if (!res.ok) throw new Error(`backend health check failed: ${res.status}`);
  return res.json();
}
