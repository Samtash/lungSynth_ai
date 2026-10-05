from __future__ import annotations

import io
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from app import storage
from app.config import settings
from app.core.jobs import create_job, get_job, run_job
from app.schemas import GenerateResponse, JobStatusResponse, PhaseResult, ResultsResponse, UploadResponse

router = APIRouter(prefix="/api", tags=["generate"])

# job_id -> session_id, and session_id -> job_id, kept alongside the job
# store so /results can be looked up either way.
_SESSION_TO_JOB: dict[str, str] = {}


def _session_dir(session_id: str) -> Path:
    d = settings.uploads_dir / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _save_upload(upload: UploadFile, dest: Path):
    filename = (upload.filename or "").lower()
    suffixes = "".join(Path(filename).suffixes)
    if suffixes and not any(filename.endswith(ext) for ext in (".dcm", ".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".zip")):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffixes}'. Accepted: DICOM, NIfTI, MetaImage, NRRD.",
        )
    with dest.open("wb") as f:
        while chunk := await upload.read(1024 * 1024):
            f.write(chunk)


@router.post("/sessions/upload", response_model=UploadResponse)
async def upload_scans(t00: UploadFile = File(...), t50: UploadFile = File(...)) -> UploadResponse:
    session_id = str(uuid.uuid4())
    session_dir = _session_dir(session_id)

    def full_suffix(filename: str | None, fallback: str) -> str:
        # Path.suffix only returns the last extension (.nii.gz -> .gz),
        # so join all suffixes to preserve compound extensions.
        suffixes = "".join(Path(filename or "").suffixes)
        return suffixes or fallback

    t00_dest = session_dir / f"t00{full_suffix(t00.filename, '.nii.gz')}"
    t50_dest = session_dir / f"t50{full_suffix(t50.filename, '.nii.gz')}"

    await _save_upload(t00, t00_dest)
    await _save_upload(t50, t50_dest)

    return UploadResponse(
        sessionId=session_id,
        t00Filename=t00.filename or t00_dest.name,
        t50Filename=t50.filename or t50_dest.name,
    )


@router.post("/sessions/{session_id}/generate", response_model=GenerateResponse)
async def start_generation(session_id: str, background_tasks: BackgroundTasks) -> GenerateResponse:
    session_dir = settings.uploads_dir / session_id
    candidates_t00 = list(session_dir.glob("t00*"))
    candidates_t50 = list(session_dir.glob("t50*"))
    if not candidates_t00 or not candidates_t50:
        raise HTTPException(status_code=404, detail="Session not found or missing uploaded scans.")

    job = create_job(session_id)
    _SESSION_TO_JOB[session_id] = job.id
    background_tasks.add_task(run_job, job.id, candidates_t00[0], candidates_t50[0])
    return GenerateResponse(jobId=job.id)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatusResponse(
        jobId=job.id, status=job.status, stage=job.stage, progress=round(job.progress, 1), error=job.error
    )


def _phase_urls(session_id: str, label: str) -> tuple[str, str]:
    return (
        f"/api/sessions/{session_id}/phases/{label}/preview",
        f"/api/sessions/{session_id}/phases/{label}/download",
    )


@router.get("/sessions/{session_id}/results", response_model=ResultsResponse)
def get_results(session_id: str) -> ResultsResponse:
    job_id = _SESSION_TO_JOB.get(session_id)
    job = get_job(job_id) if job_id else None
    if job is None or job.results is None:
        raise HTTPException(status_code=404, detail="No completed results for this session yet.")

    now = datetime.now(timezone.utc).isoformat()
    phases = []
    for p in job.results:
        preview_url, download_url = _phase_urls(session_id, p.label)
        phases.append(
            PhaseResult(
                id=p.label,
                label=p.label,
                confidence=round(p.confidence, 4),
                generationMs=p.generation_ms,
                resolution=p.resolution,
                timestamp=now,
                previewUrl=preview_url,
                downloadUrl=download_url,
            )
        )

    from app.core.inference import PhaseSynthPipeline

    return ResultsResponse(
        sessionId=session_id,
        modelVersion=settings.model_version,
        checkpointLoaded=PhaseSynthPipeline.instance().checkpoint_loaded,
        phases=phases,
    )


def _find_phase_output(session_id: str, label: str):
    job_id = _SESSION_TO_JOB.get(session_id)
    job = get_job(job_id) if job_id else None
    if job is None or job.results is None:
        raise HTTPException(status_code=404, detail="No completed results for this session yet.")
    for p in job.results:
        if p.label == label:
            return p
    raise HTTPException(status_code=404, detail=f"Phase '{label}' not found in results.")


@router.get("/sessions/{session_id}/phases/{label}/preview")
def phase_preview(session_id: str, label: str) -> Response:
    phase = _find_phase_output(session_id, label)
    return Response(content=phase.preview_png, media_type="image/png")


@router.get("/sessions/{session_id}/phases/{label}/download")
def phase_download(session_id: str, label: str) -> Response:
    phase = _find_phase_output(session_id, label)
    return Response(
        content=phase.nifti_bytes,
        media_type="application/gzip",
        headers={"Content-Disposition": f'attachment; filename="{session_id}_{label}.nii.gz"'},
    )


@router.get("/sessions/{session_id}/download-all")
def download_all(session_id: str) -> StreamingResponse:
    job_id = _SESSION_TO_JOB.get(session_id)
    job = get_job(job_id) if job_id else None
    if job is None or job.results is None:
        raise HTTPException(status_code=404, detail="No completed results for this session yet.")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in job.results:
            zf.writestr(f"{session_id}_{p.label}.nii.gz", p.nifti_bytes)
            zf.writestr(f"{session_id}_{p.label}_preview.png", p.preview_png)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="lungsynth_{session_id}.zip"'},
    )


@router.post("/sessions/{session_id}/save-history")
def save_history(session_id: str, user_email: str | None = None):
    """Called by the frontend once results finish loading, to persist a
    HistoryEntry row (mirrors addHistory() in src/lib/lungsynth.ts)."""
    job_id = _SESSION_TO_JOB.get(session_id)
    job = get_job(job_id) if job_id else None
    if job is None or job.results is None or job.started_at is None or job.finished_at is None:
        raise HTTPException(status_code=404, detail="No completed results for this session yet.")

    session_code = f"LS-{session_id[:5].upper()}-{datetime.now().year}"
    storage.add_session_record(
        record_id=str(uuid.uuid4()),
        session_code=session_code,
        user_email=user_email,
        created_at=datetime.now(timezone.utc).isoformat(),
        phases=len(job.results),
        processing_seconds=int(job.finished_at - job.started_at),
    )
    return {"ok": True}
