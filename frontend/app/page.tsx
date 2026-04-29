"use client";

import { useState } from "react";

import { AnalysisResult } from "@/components/AnalysisResult";
import { UploadCard } from "@/components/UploadCard";
import type { AnalyzeResponse } from "@/lib/types";

export default function HomePage() {
  const [result, setResult] = useState<{ data: AnalyzeResponse; video: File } | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  return (
    <main className="mx-auto min-h-screen px-6 py-12">
      <header className="mx-auto mb-10 max-w-5xl text-center">
        <span className="rounded-full border border-ink/10 bg-white px-3 py-1 text-xs uppercase tracking-widest text-ink/60">
          Qdrant Hackathon · Late-Interaction Pose Search
        </span>
        <h1 className="mt-6 text-4xl font-semibold tracking-tight md:text-5xl">
          Mimic<span className="text-accent">Coach</span>
        </h1>
        <p className="mx-auto mt-4 max-w-xl text-balance text-ink/70">
          Upload a phone clip — see it side-by-side with the closest pro,
          phase by phase, scored by Qdrant late-interaction MaxSim.
        </p>
      </header>

      {errorMsg ? (
        <div className="mx-auto mb-6 max-w-2xl rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {errorMsg}
          <button
            className="ml-2 underline-offset-2 hover:underline"
            onClick={() => setErrorMsg(null)}
          >
            dismiss
          </button>
        </div>
      ) : null}

      {result ? (
        <AnalysisResult
          result={result.data}
          userVideo={result.video}
          onReset={() => setResult(null)}
        />
      ) : (
        <UploadCard
          onResult={(r, video) => {
            setErrorMsg(null);
            setResult({ data: r, video });
          }}
          onError={(msg) => setErrorMsg(msg)}
        />
      )}

      <footer className="mx-auto mt-16 max-w-5xl text-center text-xs text-ink/50">
        <p>
          MIT · pose extraction with MediaPipe · multivector retrieval with{" "}
          <a className="underline-offset-2 hover:underline" href="https://qdrant.tech">
            Qdrant
          </a>
          .
        </p>
      </footer>
    </main>
  );
}
