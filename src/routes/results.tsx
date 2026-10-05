import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { AlertTriangle, Download, FileDown, Loader2, Maximize2, ZoomIn, ZoomOut, Move, Expand } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { CtPreview } from "@/components/CtPreview";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { getCurrentRun, downloadAllUrl, getResults, resolveAssetUrl, type ApiPhaseResult, type ApiResultsResponse } from "@/lib/api";

export const Route = createFileRoute("/results")({
  head: () => ({
    meta: [
      { title: "Generated CT Phases — LungSynth AI" },
      { name: "description", content: "Review AI-generated intermediate lung CT breathing phases T10 through T80 with confidence scores and metadata." },
      { property: "og:title", content: "Generated CT Phases — LungSynth AI" },
      { property: "og:description", content: "Inspect, zoom and export AI-reconstructed intermediate 4D lung CT phases." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Results,
});

function Results() {
  const navigate = useNavigate();
  const [data, setData] = useState<ApiResultsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<ApiPhaseResult | null>(null);
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    const run = getCurrentRun();
    if (!run) {
      navigate({ to: "/dashboard" });
      return;
    }
    getResults(run.sessionId)
      .then(setData)
      .catch((err) => setError(err instanceof Error ? err.message : "Could not load results."));
  }, [navigate]);

  if (error) {
    return (
      <AppShell>
        <div className="mx-auto flex max-w-md flex-col items-center text-center">
          <div className="flex size-16 items-center justify-center rounded-full bg-destructive/10 text-destructive">
            <AlertTriangle className="size-8" />
          </div>
          <h1 className="mt-6 text-2xl font-extrabold tracking-tight">Couldn&apos;t load results</h1>
          <p className="mt-2.5 text-sm text-muted-foreground">{error}</p>
          <Button className="mt-8 rounded-xl" onClick={() => navigate({ to: "/dashboard" })}>
            Start a new generation
          </Button>
        </div>
      </AppShell>
    );
  }

  if (!data) {
    return (
      <AppShell>
        <div className="flex min-h-[40vh] items-center justify-center">
          <Loader2 className="size-8 animate-spin text-primary" />
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell>
      <div className="animate-fade-up flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl">
            Generated Intermediate Lung CT Phases
          </h1>
          <p className="mt-2.5 text-sm text-muted-foreground sm:text-base">
            AI-generated breathing phases between T00 and T50 · {data.modelVersion}
            {!data.checkpointLoaded ? " · untrained weights (demo)" : ""}
          </p>
        </div>
        <div className="flex shrink-0 gap-2.5">
          <Button asChild variant="outline" className="rounded-xl">
            <a href={downloadAllUrl(data.sessionId)}>
              <FileDown className="size-4" /> Export Results
            </a>
          </Button>
          <Button asChild className="rounded-xl">
            <a href={downloadAllUrl(data.sessionId)}>
              <Download className="size-4" /> Download All
            </a>
          </Button>
        </div>
      </div>

      {!data.checkpointLoaded ? (
        <div className="animate-fade-up mt-6 flex items-start gap-2.5 rounded-xl border border-amber-500/30 bg-amber-500/5 px-4 py-3 text-sm text-amber-700 dark:text-amber-400">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" />
          <p>
            No trained checkpoint is loaded on the backend, so these volumes come from a real diffusion
            sampling pass through an untrained model — structurally valid, not yet clinically meaningful.
            Train the model on DIR-Lab data (see <code>backend/train.py</code>) to get real reconstructions.
          </p>
        </div>
      ) : null}

      <div className="mt-8 grid gap-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {data.phases.map((p, i) => (
          <button
            key={p.id}
            onClick={() => {
              setZoom(1);
              setOpen(p);
            }}
            style={{ animationDelay: `${i * 0.05}s` }}
            className="animate-fade-up group rounded-2xl border border-border bg-card p-4 text-left shadow-[var(--shadow-soft)] transition-all hover:-translate-y-1 hover:shadow-[var(--shadow-lift)]"
          >
            <div className="flex items-center justify-between">
              <span className="rounded-lg bg-accent px-2.5 py-1 text-xs font-bold text-accent-foreground">
                {p.label}
              </span>
              <Expand className="size-4 text-muted-foreground transition-colors group-hover:text-primary" />
            </div>
            <CtPreview className="mt-3 aspect-square w-full" label={p.label} src={resolveAssetUrl(p.previewUrl)} />
            <dl className="mt-4 space-y-1.5 text-xs">
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Confidence</dt>
                <dd className="font-semibold text-success">{(p.confidence * 100).toFixed(1)}%</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Generation time</dt>
                <dd className="font-semibold">{(p.generationMs / 1000).toFixed(2)}s</dd>
              </div>
            </dl>
          </button>
        ))}
      </div>

      <Dialog open={Boolean(open)} onOpenChange={(o) => !o && setOpen(null)}>
        <DialogContent className="max-w-5xl gap-0 overflow-hidden rounded-2xl p-0">
          <DialogTitle className="sr-only">Phase {open?.label} viewer</DialogTitle>
          {open ? (
            <div className="grid lg:grid-cols-[1fr_18rem]">
              <div className="relative bg-foreground/95 p-4">
                <div className="relative aspect-square w-full overflow-hidden rounded-xl">
                  <div
                    className="size-full origin-center transition-transform duration-200"
                    style={{ transform: `scale(${zoom})` }}
                  >
                    <CtPreview className="size-full" label={open.label} src={resolveAssetUrl(open.previewUrl)} />
                  </div>
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <Button size="sm" variant="secondary" className="rounded-lg" onClick={() => setZoom((z) => Math.min(3, z + 0.25))}>
                    <ZoomIn className="size-4" /> Zoom in
                  </Button>
                  <Button size="sm" variant="secondary" className="rounded-lg" onClick={() => setZoom((z) => Math.max(1, z - 0.25))}>
                    <ZoomOut className="size-4" /> Zoom out
                  </Button>
                  <Button size="sm" variant="secondary" className="rounded-lg">
                    <Move className="size-4" /> Pan
                  </Button>
                  <Button size="sm" variant="secondary" className="rounded-lg">
                    <Maximize2 className="size-4" /> Full screen
                  </Button>
                  <Button asChild size="sm" className="ml-auto rounded-lg">
                    <a href={resolveAssetUrl(open.downloadUrl)}>
                      <Download className="size-4" /> Download Volume
                    </a>
                  </Button>
                </div>
              </div>

              <aside className="border-t border-border bg-card p-6 lg:border-l lg:border-t-0">
                <h2 className="text-lg font-bold">Phase {open.label}</h2>
                <p className="mt-1 text-xs text-muted-foreground">Interpolated breathing phase</p>
                <dl className="mt-6 space-y-4 text-sm">
                  {[
                    ["Image resolution", open.resolution],
                    ["Generation timestamp", new Date(open.timestamp).toLocaleString()],
                    ["AI model version", data.modelVersion],
                    ["Processing duration", `${(open.generationMs / 1000).toFixed(2)}s`],
                    ["Confidence score", `${(open.confidence * 100).toFixed(1)}%`],
                  ].map(([k, v]) => (
                    <div key={k}>
                      <dt className="text-xs font-medium text-muted-foreground">{k}</dt>
                      <dd className="mt-0.5 font-semibold">{v}</dd>
                    </div>
                  ))}
                </dl>
              </aside>
            </div>
          ) : null}
        </DialogContent>
      </Dialog>
    </AppShell>
  );
}
