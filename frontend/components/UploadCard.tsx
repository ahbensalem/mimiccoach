"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";

import { fetchMotions } from "@/lib/api";
import {
  type AnalyzeResponse,
  type BodyType,
  type MotionSummary,
  type SkillLevel,
  type Sport,
  FALLBACK_MOTIONS,
} from "@/lib/types";
import { analyzeVideo } from "@/lib/api";

const SPORTS: { key: Sport; label: string }[] = [
  { key: "tennis", label: "Tennis" },
  { key: "fitness", label: "Fitness" },
  { key: "golf", label: "Golf" },
];

const SKILL_LEVELS: { key: SkillLevel; label: string }[] = [
  { key: "beginner", label: "Beginner" },
  { key: "intermediate", label: "Intermediate" },
  { key: "pro", label: "Pro" },
];

const BODY_TYPES: { key: BodyType; label: string }[] = [
  { key: "narrow", label: "Narrow" },
  { key: "balanced", label: "Balanced" },
  { key: "broad", label: "Broad" },
];

export function UploadCard({
  onResult,
  onError,
  disabled = false,
}: {
  onResult: (r: AnalyzeResponse, video: File) => void;
  onError: (msg: string) => void;
  disabled?: boolean;
}) {
  const [motions, setMotions] = useState<MotionSummary[]>(FALLBACK_MOTIONS);
  const [sport, setSport] = useState<Sport>("tennis");
  const [motionKey, setMotionKey] = useState<string>("tennis_serve");
  const [skillLevel, setSkillLevel] = useState<SkillLevel | null>(null);
  const [bodyType, setBodyType] = useState<BodyType | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Try to refresh the motions list from the backend, but don't block the UI on it.
  useEffect(() => {
    let cancelled = false;
    fetchMotions()
      .then((r) => {
        if (!cancelled && r.motions?.length) setMotions(r.motions);
      })
      .catch(() => {
        // Static fallback was already loaded; nothing to do.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const motionsBySport = useMemo(() => {
    const grouped: Record<Sport, MotionSummary[]> = {
      tennis: [],
      fitness: [],
      golf: [],
    };
    for (const m of motions) grouped[m.sport]?.push(m);
    return grouped;
  }, [motions]);

  // Snap motionKey back into the current sport when sport changes.
  useEffect(() => {
    const list = motionsBySport[sport];
    if (list?.length && !list.some((m) => m.key === motionKey)) {
      setMotionKey(list[0].key);
    }
  }, [sport, motionsBySport, motionKey]);

  function handleFile(f: File | undefined) {
    if (!f) return;
    if (!f.type.startsWith("video/") && !/\.(mp4|mov|m4v)$/i.test(f.name)) {
      onError(`Not a video file: ${f.name}`);
      return;
    }
    if (f.size > 60 * 1024 * 1024) {
      onError(`File too large (${(f.size / 1024 / 1024).toFixed(1)} MB). Keep clips under 60 MB.`);
      return;
    }
    setFile(f);
  }

  async function submit() {
    if (!file) {
      onError("Choose a video first.");
      return;
    }
    setSubmitting(true);
    try {
      const result = await analyzeVideo({
        video: file,
        motion: motionKey,
        skill_level: skillLevel ?? undefined,
        body_type: bodyType ?? undefined,
      });
      onResult(result, file);
    } catch (err) {
      onError(String(err));
    } finally {
      setSubmitting(false);
    }
  }

  const isBusy = submitting || disabled;

  return (
    <section className="mx-auto w-full max-w-2xl space-y-6 rounded-3xl border border-ink/10 bg-white p-6 shadow-sm">
      <div className="space-y-1">
        <h2 className="text-lg font-semibold">Analyze your motion</h2>
        <p className="text-sm text-ink/60">
          Upload a phone clip (MP4 / MOV, ≤ 60 MB). Side angle works best.
        </p>
      </div>

      <fieldset className="space-y-2" disabled={isBusy}>
        <legend className="text-xs uppercase tracking-widest text-ink/50">Sport</legend>
        <div className="flex flex-wrap gap-2">
          {SPORTS.map((s) => (
            <button
              key={s.key}
              type="button"
              onClick={() => setSport(s.key)}
              className={chipClass(sport === s.key)}
            >
              {s.label}
            </button>
          ))}
        </div>
      </fieldset>

      <fieldset className="space-y-2" disabled={isBusy}>
        <legend className="text-xs uppercase tracking-widest text-ink/50">Motion</legend>
        <div className="flex flex-wrap gap-2">
          {motionsBySport[sport].map((m) => (
            <button
              key={m.key}
              type="button"
              onClick={() => setMotionKey(m.key)}
              className={chipClass(motionKey === m.key)}
              title={m.is_hero ? "Hero motion — most polished" : ""}
            >
              {m.label}
              {m.is_hero ? <span className="ml-1.5 text-amber" aria-hidden>★</span> : null}
            </button>
          ))}
        </div>
      </fieldset>

      <fieldset className="space-y-2" disabled={isBusy}>
        <legend className="text-xs uppercase tracking-widest text-ink/50">
          Filters <span className="text-ink/40">(optional)</span>
        </legend>
        <div className="space-y-2">
          <FilterRow
            label="Skill"
            options={SKILL_LEVELS}
            value={skillLevel}
            onChange={setSkillLevel}
          />
          <FilterRow
            label="Body type"
            options={BODY_TYPES}
            value={bodyType}
            onChange={setBodyType}
          />
        </div>
      </fieldset>

      <div
        onDragEnter={(e) => {
          e.preventDefault();
          if (!isBusy) setDragActive(true);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!isBusy) setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragActive(false);
          if (isBusy) return;
          handleFile(e.dataTransfer.files?.[0]);
        }}
        className={clsx(
          "rounded-2xl border-2 border-dashed transition-colors",
          dragActive ? "border-accent bg-accent/5" : "border-ink/15 bg-canvas",
          "p-6 text-center",
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/mp4,video/quicktime,.mp4,.mov,.m4v"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0] ?? undefined)}
        />
        {file ? (
          <div className="space-y-2">
            <p className="text-sm font-medium">{file.name}</p>
            <p className="text-xs text-ink/60">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
            <button
              type="button"
              className="text-xs text-accent underline-offset-2 hover:underline"
              onClick={() => {
                setFile(null);
                if (inputRef.current) inputRef.current.value = "";
              }}
            >
              Choose a different file
            </button>
          </div>
        ) : (
          <>
            <p className="text-sm text-ink/70">Drag &amp; drop your video here</p>
            <p className="my-2 text-xs uppercase tracking-widest text-ink/40">or</p>
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              disabled={isBusy}
              className="rounded-full bg-ink px-4 py-2 text-sm font-medium text-canvas disabled:opacity-50"
            >
              Browse files
            </button>
          </>
        )}
      </div>

      <button
        type="button"
        onClick={submit}
        disabled={isBusy || !file}
        className={clsx(
          "w-full rounded-full py-3 text-sm font-semibold transition-colors",
          file && !isBusy
            ? "bg-accent text-canvas hover:bg-accent/90"
            : "bg-ink/10 text-ink/40",
        )}
      >
        {submitting ? "Analyzing…" : "Analyze"}
      </button>
    </section>
  );
}

function FilterRow<T extends string>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: { key: T; label: string }[];
  value: T | null;
  onChange: (v: T | null) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-24 shrink-0 text-xs text-ink/60">{label}</span>
      <div className="flex flex-wrap gap-2">
        <button type="button" onClick={() => onChange(null)} className={chipClass(value === null, "sm")}>
          Any
        </button>
        {options.map((o) => (
          <button
            key={o.key}
            type="button"
            onClick={() => onChange(o.key)}
            className={chipClass(value === o.key, "sm")}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function chipClass(active: boolean, size: "md" | "sm" = "md"): string {
  return clsx(
    "rounded-full border transition-colors",
    size === "sm" ? "px-3 py-1 text-xs" : "px-4 py-1.5 text-sm",
    active
      ? "border-ink bg-ink text-canvas"
      : "border-ink/15 bg-white text-ink hover:bg-ink/5",
  );
}
