import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { AlertTriangle, Check, Loader2 } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { Button } from "@/components/ui/button";
import { getUser } from "@/lib/lungsynth";
import { getCurrentRun, getJobStatus, saveHistory, type JobStatus } from "@/lib/api";

export const Route = createFileRoute("/processing")({
  head: () => ({
    meta: [
      { title: "Processing — LungSynth AI" },
      { name: "description", content: "The phase-conditioned diffusion model is reconstructing the intermediate lung CT breathing cycle." },
      { property: "og:title", content: "Processing — LungSynth AI" },
      { property: "og:description", content: "Live progress for AI reconstruction of intermediate 4D lung CT phases." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Processing,
});

const STAGES = [
  "Uploading scans",
  "Preprocessing",
  "Running Diffusion Model",
  "Generating CT Phases",
  "Finalizing Results",
];

const POLL_MS = 800;

function stageIndex(stage: string) {
  const i = STAGES.indexOf(stage);
  return i === -1 ? 0 : i;
}

function Processing() {
  const navigate = useNavigate();
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const startedAt = useRef(Date.now());

  useEffect(() => {
    const run = getCurrentRun();
    if (!run) {
      navigate({ to: "/dashboard" });
      return;
    }

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const poll = async () => {
      try {
        const js = await getJobStatus(run.jobId);
        if (cancelled) return;
        setStatus(js);

        if (js.status === "completed") {
          const user = getUser();
          try {
            await saveHistory(run.sessionId, user?.email);
          } catch {
            // History persistence failing shouldn't block viewing results.
          }
          navigate({ to: "/results" });
          return;
        }
        if (js.status === "failed") {
          setError(js.error ?? "Generation failed on the backend.");
          return;
        }
        timer = setTimeout(poll, POLL_MS);
      } catch (err) {
        if (cancelled) return;
        setError(
          err instanceof Error
            ? `Lost contact with the backend: ${err.message}`
            : "Lost contact with the backend.",
        );
      }
    };

    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [navigate]);

  if (error) {
    return (
      <AppShell>
        <div className="mx-auto flex max-w-md flex-col items-center text-center">
          <div className="flex size-16 items-center justify-center rounded-full bg-destructive/10 text-destructive">
            <AlertTriangle className="size-8" />
          </div>
          <h1 className="mt-6 text-2xl font-extrabold tracking-tight">Generation failed</h1>
          <p className="mt-2.5 whitespace-pre-wrap break-words text-left text-xs text-muted-foreground">{error}</p>
          <Button className="mt-8 rounded-xl" onClick={() => navigate({ to: "/dashboard" })}>
            Back to Upload
          </Button>
        </div>
      </AppShell>
    );
  }

  const progress = status?.progress ?? 2;
  const activeStage = status ? stageIndex(status.stage) : 0;
  const elapsedS = Math.max(1, Math.round((Date.now() - startedAt.current) / 1000));

  return (
    <AppShell>
      <div className="mx-auto flex max-w-2xl flex-col items-center text-center">
        <div className="animate-fade-up relative size-44">
          <div className="absolute inset-0 animate-ct-spin rounded-full border-2 border-dashed border-primary/25" />
          <div className="absolute inset-3 animate-ct-spin rounded-full border-2 border-secondary/30 [animation-direction:reverse] [animation-duration:5s]" />
          <div className="absolute inset-8 overflow-hidden rounded-full">
            <div className="ct-surface size-full" />
            <div className="absolute inset-x-0 h-1/3 animate-scan-sweep bg-[linear-gradient(to_bottom,transparent,rgba(255,255,255,0.16),transparent)]" />
          </div>
          <div className="absolute inset-0 animate-ct-spin [animation-duration:2.4s]">
            <span className="absolute left-1/2 top-0 size-2.5 -translate-x-1/2 rounded-full bg-secondary" />
          </div>
        </div>

        <h1 className="animate-fade-up mt-10 text-2xl font-extrabold tracking-tight sm:text-3xl">
          Generating Intermediate CT Phases
        </h1>
        <p className="animate-fade-up mt-2.5 text-sm text-muted-foreground">
          Please wait while the AI reconstructs the breathing cycle.
        </p>

        <div className="mt-10 w-full rounded-2xl border border-border bg-card p-6 text-left shadow-[var(--shadow-soft)]">
          <ul className="flex flex-col gap-4">
            {STAGES.map((s, i) => {
              const done = i < activeStage;
              const active = i === activeStage;
              return (
                <li key={s} className="flex items-center gap-3">
                  <span
                    className={`flex size-7 items-center justify-center rounded-full border transition-colors ${
                      done
                        ? "border-success bg-success text-success-foreground"
                        : active
                          ? "border-primary bg-accent text-primary"
                          : "border-border bg-muted text-muted-foreground"
                    }`}
                  >
                    {done ? (
                      <Check className="size-4" />
                    ) : active ? (
                      <Loader2 className="size-4 animate-spin" />
                    ) : (
                      <span className="text-[11px] font-bold">{i + 1}</span>
                    )}
                  </span>
                  <div className="flex-1">
                    <p className={`text-sm font-semibold ${done || active ? "text-foreground" : "text-muted-foreground"}`}>
                      Stage {i + 1} · {s}
                    </p>
                  </div>
                </li>
              );
            })}
          </ul>

          <div className="mt-6">
            <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-primary transition-[width] duration-200 ease-linear"
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="mt-2.5 flex justify-between text-xs font-medium text-muted-foreground">
              <span>{Math.round(progress)}% complete</span>
              <span>Elapsed: {elapsedS}s</span>
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
