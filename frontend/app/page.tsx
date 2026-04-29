export default function HomePage() {
  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col items-center justify-center px-6 py-16">
      <span className="rounded-full border border-ink/10 bg-white px-3 py-1 text-xs uppercase tracking-widest text-ink/60">
        Qdrant Hackathon Submission
      </span>

      <h1 className="mt-8 text-balance text-center text-5xl font-semibold tracking-tight md:text-6xl">
        Mimic<span className="text-accent">Coach</span>
      </h1>

      <p className="mt-6 max-w-xl text-balance text-center text-lg text-ink/70">
        Upload a phone clip of your tennis serve, squat, or golf swing.
        See it side-by-side with the closest pro — phase by phase, scored by{" "}
        <span className="font-medium text-accent">Qdrant late-interaction MaxSim</span>.
      </p>

      <div className="mt-12 grid w-full grid-cols-1 gap-3 sm:grid-cols-3">
        {[
          { sport: "Tennis", motions: "Serve · Forehand · Backhand" },
          { sport: "Fitness", motions: "Squat · Bench · Row" },
          { sport: "Golf", motions: "Full swing" },
        ].map((s) => (
          <div
            key={s.sport}
            className="rounded-2xl border border-ink/10 bg-white px-4 py-3 text-center text-sm"
          >
            <div className="font-medium">{s.sport}</div>
            <div className="text-ink/60">{s.motions}</div>
          </div>
        ))}
      </div>

      <button
        type="button"
        disabled
        className="mt-12 rounded-full bg-ink px-6 py-3 text-sm font-medium text-canvas opacity-60"
        aria-disabled
      >
        Upload coming in P6 — backend wires up first
      </button>
    </main>
  );
}
