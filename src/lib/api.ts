/**
 * Client for the real LungSynth AI backend (backend/, FastAPI).
 * Replaces the localStorage-mocked generation flow in lungsynth.ts with
 * actual HTTP calls: upload -> generate -> poll -> results.
 */

export const API_BASE_URL =
  (import.meta as unknown as { env?: Record<string, string> }).env?.VITE_API_BASE_URL ??
  "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // ignore body parse failure, fall back to statusText
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return (await res.json()) as T;
}

export type UploadResponse = {
  sessionId: string;
  t00Filename: string;
  t50Filename: string;
};

export async function uploadScans(t00: File, t50: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("t00", t00);
  form.append("t50", t50);
  return request<UploadResponse>("/api/sessions/upload", { method: "POST", body: form });
}

export async function startGeneration(sessionId: string): Promise<{ jobId: string }> {
  return request(`/api/sessions/${sessionId}/generate`, { method: "POST" });
}

export type JobStatus = {
  jobId: string;
  status: "queued" | "running" | "completed" | "failed";
  stage: string;
  progress: number;
  error: string | null;
};

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  return request(`/api/jobs/${jobId}`);
}

export type ApiPhaseResult = {
  id: string;
  label: string;
  confidence: number;
  generationMs: number;
  resolution: string;
  timestamp: string;
  previewUrl: string;
  downloadUrl: string;
};

export type ApiResultsResponse = {
  sessionId: string;
  modelVersion: string;
  checkpointLoaded: boolean;
  phases: ApiPhaseResult[];
};

export async function getResults(sessionId: string): Promise<ApiResultsResponse> {
  return request(`/api/sessions/${sessionId}/results`);
}

export function resolveAssetUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

export function downloadAllUrl(sessionId: string): string {
  return `${API_BASE_URL}/api/sessions/${sessionId}/download-all`;
}

export async function saveHistory(sessionId: string, userEmail?: string): Promise<void> {
  const params = userEmail ? `?user_email=${encodeURIComponent(userEmail)}` : "";
  await request(`/api/sessions/${sessionId}/save-history${params}`, { method: "POST" });
}

export type ApiHistoryEntry = {
  id: string;
  sessionId: string;
  createdAt: string;
  phases: number;
  processingSeconds: number;
  status: string;
};

export async function fetchHistory(userEmail?: string): Promise<ApiHistoryEntry[]> {
  const params = userEmail ? `?user_email=${encodeURIComponent(userEmail)}` : "";
  return request(`/api/history${params}`);
}

export async function clearHistoryRemote(userEmail?: string): Promise<void> {
  const params = userEmail ? `?user_email=${encodeURIComponent(userEmail)}` : "";
  await request(`/api/history${params}`, { method: "DELETE" });
}

// Transient per-tab handoff between /dashboard -> /processing -> /results.
// sessionStorage (not localStorage) because it's only meaningful for the
// generation run currently in flight.
const CURRENT_RUN_KEY = "lungsynth.currentRun";

export type CurrentRun = { sessionId: string; jobId: string };

export function setCurrentRun(run: CurrentRun) {
  window.sessionStorage.setItem(CURRENT_RUN_KEY, JSON.stringify(run));
}

export function getCurrentRun(): CurrentRun | null {
  const raw = window.sessionStorage.getItem(CURRENT_RUN_KEY);
  return raw ? (JSON.parse(raw) as CurrentRun) : null;
}

export async function checkHealth(): Promise<{
  status: string;
  device: string;
  checkpointLoaded: boolean;
  modelVersion: string;
}> {
  return request("/api/health");
}
