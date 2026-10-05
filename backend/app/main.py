from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import storage
from app.config import settings
from app.routers import auth, generate, history

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="LungSynth AI Backend",
    description=(
        "Phase-conditioned diffusion model backend for 4D lung CT intermediate "
        "phase synthesis (3D U-Net + FiLM + Phase-Aware Attention, DDPM/DDIM)."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    storage.init_db()
    # Warm-load the model once at startup instead of on first request so
    # the first user doesn't pay the model-construction/checkpoint-load cost.
    from app.core.inference import PhaseSynthPipeline

    pipeline = PhaseSynthPipeline.instance()
    logging.getLogger("lungsynth").info(
        "Model ready on device=%s | checkpoint_loaded=%s | checkpoint_path=%s",
        pipeline.device,
        pipeline.checkpoint_loaded,
        settings.checkpoint_path,
    )
    if not pipeline.checkpoint_loaded:
        logging.getLogger("lungsynth").warning(
            "No trained checkpoint found at %s — running with randomly "
            "initialised weights. Outputs will be structurally valid CT "
            "volumes but not clinically meaningful. See backend/train.py.",
            settings.checkpoint_path,
        )


app.include_router(auth.router)
app.include_router(generate.router)
app.include_router(history.router)


@app.get("/api/health")
def health():
    from app.core.inference import PhaseSynthPipeline

    pipeline = PhaseSynthPipeline.instance()
    return {
        "status": "ok",
        "device": pipeline.device,
        "checkpointLoaded": pipeline.checkpoint_loaded,
        "modelVersion": settings.model_version,
    }
