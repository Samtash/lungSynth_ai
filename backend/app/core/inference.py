"""
End-to-end inference pipeline: two boundary CT scans in -> seven
intermediate-phase volumes out, using the real 3D U-Net + FiLM +
Phase-Aware Attention diffusion model (not a placeholder).

IMPORTANT — trained weights: this repository does not ship a trained
checkpoint (the source proposal is a *research proposal*; the DIR-Lab
model has not been trained yet, see backend/train.py). If
`checkpoints/phasediff.pt` is absent, the pipeline runs with randomly
initialised weights. The forward pass, conditioning and DDIM sampling
are all real and will produce a correctly-shaped CT-like volume, but
the *content* of that volume is not clinically meaningful until you
train the model (see backend/train.py) and drop the resulting
checkpoint at `checkpoints/phasediff.pt`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from app.config import settings
from app.core.diffusion import GaussianDiffusion
from app.core.preprocessing import (
    load_volume,
    mid_slice_preview_png,
    resample_to_shape,
    tensor_to_nifti_bytes,
    volume_to_tensor,
)
from app.models.unet3d import UNet3D


def _resolve_device() -> str:
    if settings.device != "auto":
        return settings.device
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PhaseOutput:
    label: str
    phase_fraction: float
    generation_ms: int
    confidence: float
    resolution: str
    preview_png: bytes
    nifti_bytes: bytes


class PhaseSynthPipeline:
    """Loads the model + diffusion schedule once and reuses them across
    requests. Not thread-safe for concurrent generate() calls on the
    same process by design -- the job queue in core/jobs.py serialises
    generation requests instead of adding model-level locking."""

    _instance: "PhaseSynthPipeline | None" = None

    def __init__(self):
        self.device = _resolve_device()
        self.model = UNet3D(
            in_channels=3,
            out_channels=1,
            base_channels=settings.unet_base_channels,
            channel_mults=settings.unet_channel_mults,
        ).to(self.device)
        self.model.eval()

        self.diffusion = GaussianDiffusion(
            timesteps=settings.diffusion_timesteps, device=self.device
        )

        self.checkpoint_loaded = False
        self._maybe_load_checkpoint()

    @classmethod
    def instance(cls) -> "PhaseSynthPipeline":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _maybe_load_checkpoint(self):
        ckpt_path = settings.checkpoint_path
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=self.device)
            state_dict = state.get("model_state_dict", state)
            self.model.load_state_dict(state_dict)
            self.checkpoint_loaded = True

    @torch.no_grad()
    def generate(
        self,
        t00_path: str | Path,
        t50_path: str | Path,
        phase_labels: tuple[str, ...] | None = None,
        progress_cb: Callable[[str, float], None] | None = None,
    ) -> list[PhaseOutput]:
        """Generate every requested intermediate phase from the two
        boundary scans. progress_cb(stage_name, fraction_0_to_1) is
        called throughout so callers (the job queue) can report live
        progress to the frontend."""
        phase_labels = phase_labels or settings.phase_labels

        if progress_cb:
            progress_cb("Preprocessing", 0.0)

        t00_sitk = load_volume(t00_path)
        t50_sitk = load_volume(t50_path)
        t00_tensor = volume_to_tensor(t00_sitk, settings.working_size).to(self.device)
        t50_tensor = volume_to_tensor(t50_sitk, settings.working_size).to(self.device)
        reference_geometry = resample_to_shape(
            t00_sitk, (settings.working_size,) * 3
        )

        if progress_cb:
            progress_cb("Preprocessing", 1.0)

        outputs: list[PhaseOutput] = []
        n_phases = len(phase_labels)

        for idx, label in enumerate(phase_labels):
            phase_fraction = int(label[1:]) / 100.0
            start = time.perf_counter()

            def model_fn(x_t: torch.Tensor, t: torch.Tensor, _pf=phase_fraction) -> torch.Tensor:
                pf_batch = torch.full((x_t.shape[0],), _pf, device=self.device)
                return self.model(x_t, t00_tensor, t50_tensor, t, pf_batch)

            def step_progress(step: int, total: int, _i=idx):
                if progress_cb:
                    overall = (_i + step / total) / n_phases
                    progress_cb("Running Diffusion Model", overall)

            shape = t00_tensor.shape
            generated = self.diffusion.ddim_sample(
                model_fn,
                shape=shape,
                num_steps=settings.ddim_steps,
                device=self.device,
                progress_cb=step_progress,
            )

            elapsed_ms = int((time.perf_counter() - start) * 1000)

            confidence = self._consistency_score(generated, t00_tensor, t50_tensor, phase_fraction)

            preview = mid_slice_preview_png(generated, label=label)
            nifti = tensor_to_nifti_bytes(generated, reference=reference_geometry)

            outputs.append(
                PhaseOutput(
                    label=label,
                    phase_fraction=phase_fraction,
                    generation_ms=elapsed_ms,
                    confidence=confidence,
                    resolution=f"{settings.working_size}\u00b3 (resampled)",
                    preview_png=preview,
                    nifti_bytes=nifti,
                )
            )

            if progress_cb:
                progress_cb("Generating CT Phases", (idx + 1) / n_phases)

        return outputs

    @staticmethod
    def _consistency_score(
        generated: torch.Tensor, t00: torch.Tensor, t50: torch.Tensor, phase_fraction: float
    ) -> float:
        """Heuristic reconstruction-consistency score in [0, 1]: how close
        the generated volume is to the linear-interpolation baseline
        between the two boundary scans, as a sanity signal.

        This is NOT a calibrated clinical confidence value -- that would
        require Target Registration Error against DIR-Lab landmark
        ground truth (see Section 5.1.4 of the proposal / backend/train.py's
        evaluation hooks) on a *trained* model. Until the model is
        trained, treat this purely as a smoke-test metric.
        """
        baseline = (1 - phase_fraction) * t00 + phase_fraction * t50
        mae = torch.mean(torch.abs(generated - baseline)).item()
        score = 1.0 - min(mae / 2.0, 1.0)  # values are in [-1,1], so max diff is 2
        return float(np.clip(0.5 + 0.5 * score, 0.0, 0.999))
