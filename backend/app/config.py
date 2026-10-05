"""
Central configuration for the LungSynth AI backend.

All values can be overridden with environment variables (or a `.env` file
in the `backend/` directory) so the same code runs in dev and on a real
GPU box without edits.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="LUNGSYNTH_", extra="ignore")

    # --- Paths -------------------------------------------------------
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data"
    uploads_dir: Path = base_dir / "data" / "uploads"
    outputs_dir: Path = base_dir / "data" / "outputs"
    db_path: Path = base_dir / "data" / "lungsynth.db"
    checkpoint_path: Path = base_dir / "checkpoints" / "phasediff.pt"

    # --- Model / inference --------------------------------------------
    # Working resolution the whole pipeline resamples volumes to before
    # running the diffusion model. 96^3 is a practical middle ground
    # between the paper's 64^3 training patches and full 512x512x~120
    # clinical volumes -- large enough to look like a real reconstruction,
    # small enough to run diffusion sampling on a CPU in a demo setting.
    working_size: int = 96
    unet_base_channels: int = 32
    unet_channel_mults: tuple[int, ...] = (1, 2, 4, 8)
    diffusion_timesteps: int = 1000
    ddim_steps: int = 50
    device: str = "cuda" if os.environ.get("LUNGSYNTH_FORCE_CUDA") else "auto"

    # Phases the app reconstructs (matches PHASE_LABELS in the frontend).
    phase_labels: tuple[str, ...] = ("T10", "T20", "T30", "T40", "T60", "T70", "T80")

    # --- Auth -----------------------------------------------------------
    # Set this to your real OAuth client ID to enable verified Google
    # Sign-In. When unset, the backend runs in DEV mode and trusts the
    # decoded (but *unverified*) token payload -- fine for local dev,
    # never for production.
    google_client_id: str | None = None

    # --- CORS -------------------------------------------------------
    cors_origins: tuple[str, ...] = (
        "http://localhost:8080",
        "http://localhost:3000",
        "http://127.0.0.1:8080",
    )

    model_version: str = "PhaseDiff v2.4.1"


settings = Settings()

for d in (settings.data_dir, settings.uploads_dir, settings.outputs_dir, settings.checkpoint_path.parent):
    d.mkdir(parents=True, exist_ok=True)
