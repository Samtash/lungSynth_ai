"""
Training script for the phase-conditioned diffusion model on the
DIR-Lab 4D-CT dataset, following Section 4.3.4 of the proposal:
patch-based training (64^3 patches, 50% overlap stride), AdamW
(lr=2e-4, cosine annealing), 200 epochs, Leave-One-Out
Cross-Validation across the 10 DIR-Lab patients.

This is NOT run automatically by the web app. It's a standalone
script for producing the trained checkpoint the inference pipeline
loads from `checkpoints/phasediff.pt`. You need the DIR-Lab dataset
(https://www.dir-lab.com) downloaded locally first.

Expected data layout (adjust `--data-dir` / `discover_patients` below
to match whatever layout you actually download):

    data/dirlab/
      Case1Pack/
        Images/case1_T00.img (or .mhd/.raw)
        Images/case1_T10.img
        ...
        Images/case1_T90.img
      Case2Pack/
        ...

Usage:
    python train.py --data-dir data/dirlab --epochs 200 --out checkpoints/phasediff.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from app.config import settings
from app.core.diffusion import GaussianDiffusion
from app.core.preprocessing import hu_normalize, load_volume
from app.models.unet3d import UNet3D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lungsynth.train")

PATCH_SIZE = 64
STRIDE = 32
ALL_PHASES = [f"T{p:02d}" for p in range(0, 100, 10)]  # T00..T90


def discover_patients(data_dir: Path) -> list[dict]:
    """Returns a list of {patient_id, phase_paths: {label: path}} dicts.
    Adjust the glob patterns below to match your actual DIR-Lab download
    layout -- filenames vary by source/mirror."""
    patients = []
    for patient_dir in sorted(data_dir.glob("Case*")):
        phase_paths = {}
        for label in ALL_PHASES:
            matches = list(patient_dir.rglob(f"*{label}*.mhd")) or list(
                patient_dir.rglob(f"*{label}*.nii*")
            )
            if matches:
                phase_paths[label] = matches[0]
        if len(phase_paths) == len(ALL_PHASES):
            patients.append({"patient_id": patient_dir.name, "phase_paths": phase_paths})
        else:
            logger.warning("Skipping %s: only found %d/%d phases", patient_dir.name, len(phase_paths), len(ALL_PHASES))
    return patients


class DirLabPatchDataset(Dataset):
    """Extracts overlapping 64^3 patches from every intermediate phase of
    the given patients, paired with that patient's T00/T50 anchors, for
    diffusion training (Section 4.1 preprocessing, 4.3.4 patch config)."""

    def __init__(self, patients: list[dict], working_size: int = 160):
        self.samples = []  # list of (patient, target_label, patch_origin)
        self.volumes: dict[str, dict[str, np.ndarray]] = {}
        self.working_size = working_size

        for patient in patients:
            pid = patient["patient_id"]
            self.volumes[pid] = {}
            for label, path in patient["phase_paths"].items():
                sitk_img = load_volume(path)
                arr = self._to_fixed_array(sitk_img)
                self.volumes[pid][label] = hu_normalize(arr)

            for label in ALL_PHASES:
                if label in ("T00", "T50"):
                    continue  # only intermediate phases are generation targets
                for z in range(0, working_size - PATCH_SIZE + 1, STRIDE):
                    for y in range(0, working_size - PATCH_SIZE + 1, STRIDE):
                        for x in range(0, working_size - PATCH_SIZE + 1, STRIDE):
                            self.samples.append((pid, label, (z, y, x)))

    def _to_fixed_array(self, sitk_img) -> np.ndarray:
        import SimpleITK as sitk

        resampler = sitk.ResampleImageFilter()
        size = (self.working_size,) * 3
        original_size = sitk_img.GetSize()
        original_spacing = sitk_img.GetSpacing()
        new_spacing = [original_spacing[i] * (original_size[i] / size[i]) for i in range(3)]
        resampler.SetSize(size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputOrigin(sitk_img.GetOrigin())
        resampler.SetOutputDirection(sitk_img.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1000.0)
        resampled = resampler.Execute(sitk_img)
        return sitk.GetArrayFromImage(resampled).astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, label, (z, y, x) = self.samples[idx]
        s = slice(z, z + PATCH_SIZE)
        sy = slice(y, y + PATCH_SIZE)
        sx = slice(x, x + PATCH_SIZE)

        target = self.volumes[pid][label][s, sy, sx]
        t00 = self.volumes[pid]["T00"][s, sy, sx]
        t50 = self.volumes[pid]["T50"][s, sy, sx]
        phase_fraction = int(label[1:]) / 100.0

        return (
            torch.from_numpy(target).unsqueeze(0),
            torch.from_numpy(t00).unsqueeze(0),
            torch.from_numpy(t50).unsqueeze(0),
            torch.tensor(phase_fraction, dtype=torch.float32),
        )


def train_one_fold(train_patients: list[dict], held_out_id: str, args) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Fold held-out=%s | device=%s | train patients=%d", held_out_id, device, len(train_patients))

    dataset = DirLabPatchDataset(train_patients, working_size=args.working_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = UNet3D(
        in_channels=3, out_channels=1,
        base_channels=settings.unet_base_channels,
        channel_mults=settings.unet_channel_mults,
    ).to(device)
    diffusion = GaussianDiffusion(timesteps=settings.diffusion_timesteps, device=device)

    optimizer = AdamW(model.parameters(), lr=2e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    ckpt_path = args.out.parent / f"phasediff_fold_{held_out_id}.pt"

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for target, t00, t50, phase_fraction in loader:
            target, t00, t50 = target.to(device), t00.to(device), t50.to(device)
            phase_fraction = phase_fraction.to(device)

            def model_fn(x_t, t, _t00=t00, _t50=t50, _pf=phase_fraction):
                return model(x_t, _t00, _t50, t, _pf)

            loss = diffusion.training_loss(model_fn, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(1, len(loader))
        logger.info("Fold %s | epoch %d/%d | loss=%.5f", held_out_id, epoch + 1, args.epochs, avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "loss": avg_loss}, ckpt_path)

    return ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/dirlab"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/phasediff.pt"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--working-size", type=int, default=160)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--loocv", action="store_true",
        help="Run full 10-fold Leave-One-Out CV (Section 5.1.5). Without this "
             "flag, trains a single model on all available patients.",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    patients = discover_patients(args.data_dir)
    if not patients:
        raise SystemExit(
            f"No usable patients found under {args.data_dir}. Download the "
            f"DIR-Lab 4D-CT dataset (https://www.dir-lab.com) and check the "
            f"layout expected in discover_patients()."
        )
    logger.info("Discovered %d patients with all 10 phases.", len(patients))

    if args.loocv:
        for held_out in patients:
            train_set = [p for p in patients if p["patient_id"] != held_out["patient_id"]]
            train_one_fold(train_set, held_out["patient_id"], args)
        logger.info("LOOCV complete. Pick/ensemble a fold checkpoint and copy it to %s", args.out)
    else:
        ckpt = train_one_fold(patients, "all", args)
        ckpt.rename(args.out)
        logger.info("Training complete. Checkpoint saved to %s", args.out)


if __name__ == "__main__":
    main()
