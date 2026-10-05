from __future__ import annotations

from pydantic import BaseModel


class UploadResponse(BaseModel):
    sessionId: str
    t00Filename: str
    t50Filename: str


class GenerateResponse(BaseModel):
    jobId: str


class JobStatusResponse(BaseModel):
    jobId: str
    status: str  # queued | running | completed | failed
    stage: str
    progress: float
    error: str | None = None


class PhaseResult(BaseModel):
    """Mirrors PhaseResult in src/lib/lungsynth.ts exactly, so the
    frontend's existing rendering code needs minimal changes."""

    id: str
    label: str
    confidence: float
    generationMs: int
    resolution: str
    timestamp: str
    previewUrl: str
    downloadUrl: str


class ResultsResponse(BaseModel):
    sessionId: str
    modelVersion: str
    checkpointLoaded: bool
    phases: list[PhaseResult]


class HistoryEntry(BaseModel):
    id: str
    sessionId: str
    createdAt: str
    phases: int
    processingSeconds: int
    status: str


class GoogleAuthRequest(BaseModel):
    credential: str


class StoredUser(BaseModel):
    name: str
    email: str
    initials: str
