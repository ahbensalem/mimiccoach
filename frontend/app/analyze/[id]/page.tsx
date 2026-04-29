// Stub — full implementation lands in P6 (frontend MVP & overlay).
// Will host: <SplitVideo>, <SkeletonCanvas>, <PhaseScores>, <CoachingTip>.

type Params = Promise<{ id: string }>;

export default async function AnalyzePage({ params }: { params: Params }) {
  const { id } = await params;
  return (
    <main className="mx-auto max-w-4xl px-6 py-16">
      <h1 className="text-2xl font-semibold">Analysis #{id}</h1>
      <p className="mt-4 text-ink/70">Result page lands in P6.</p>
    </main>
  );
}
