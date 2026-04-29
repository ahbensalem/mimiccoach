// Stub — full component lands in P6.
// Two <video> elements with synced currentTime + a shared scrubber.
"use client";

export function SplitVideo() {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      <div className="aspect-video rounded-xl bg-ink/5" />
      <div className="aspect-video rounded-xl bg-ink/5" />
    </div>
  );
}
