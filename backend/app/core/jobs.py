"""
Lightweight in-process job tracker. A "job" is one generation run
(upload -> preprocess -> diffusion -> stitched results). Kept in memory
for simplicity -- swap for Redis/DB-backed jobs if you need multiple
worker processes.

Stage names intentionally match the frontend's STAGES array in
src/routes/processing.tsx so the progress bar reflects real work.
"""

from __future__ import annotations

import asyncio
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings
from app.core.inference import PhaseOutput, PhaseSynthPipeline

STAGES = [
    "Uploading scans",
    "Preprocessing",
    "Running Diffusion Model",
    "Generating CT Phases",
    "Finalizing Results",
]


@dataclass
class Job:
    id: str
    session_id: str
    status: str = "queued"  # queued | running | completed | failed
    stage: str = STAGES[0]
    progress: float = 0.0  # 0..100
    error: str | None = None
    results: list[PhaseOutput] | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None


_JOBS: dict[str, Job] = {}
_LOCK = asyncio.Lock()


def create_job(session_id: str) -> Job:
    job = Job(id=str(uuid.uuid4()), session_id=session_id)
    _JOBS[job.id] = job
    return job


def get_job(job_id: str) -> Job | None:
    return _JOBS.get(job_id)


async def run_job(job_id: str, t00_path: Path, t50_path: Path):
    job = _JOBS[job_id]
    job.status = "running"
    job.started_at = time.time()
    job.stage = "Uploading scans"
    job.progress = 5.0

    def progress_cb(stage: str, fraction: float):
        job.stage = stage
        stage_idx = STAGES.index(stage) if stage in STAGES else 1
        base = (stage_idx / len(STAGES)) * 100
        span = 100 / len(STAGES)
        job.progress = min(99.0, base + fraction * span)

    try:
        loop = asyncio.get_event_loop()
        pipeline = PhaseSynthPipeline.instance()

        # Run the (blocking, CPU/GPU-bound) diffusion sampling in a
        # worker thread so the FastAPI event loop stays responsive for
        # status polling from the frontend.
        results = await loop.run_in_executor(
            None,
            lambda: pipeline.generate(
                t00_path, t50_path, phase_labels=settings.phase_labels, progress_cb=progress_cb
            ),
        )

        job.stage = "Finalizing Results"
        job.progress = 100.0
        job.results = results
        job.status = "completed"
        job.finished_at = time.time()
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        job.error = f"{exc}\n{traceback.format_exc()}"
        job.finished_at = time.time()
