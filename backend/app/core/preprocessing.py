"""
Loading, HU-normalising, resampling and preview-rendering for the CT
volume formats the frontend accepts: DICOM (.dcm / a DICOM series
folder), NIfTI (.nii/.nii.gz), MetaImage (.mha/.mhd) and NRRD (.nrrd).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image

# Lung-window HU clip range used before normalising to [-1, 1] (Section 4.1 step 1).
HU_MIN, HU_MAX = -1000.0, 1000.0


def load_volume(path: str | Path) -> sitk.Image:
    """Load any supported CT volume format into a SimpleITK image.

    `path` may be a single file (.nii/.nii.gz/.mha/.mhd/.nrrd/.dcm) or a
    directory containing a DICOM series.
    """
    path = Path(path)

    if path.is_dir():
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(path))
        if not series_ids:
            raise ValueError(f"No DICOM series found in directory: {path}")
        filenames = reader.GetGDCMSeriesFileNames(str(path), series_ids[0])
        reader.SetFileNames(filenames)
        return reader.Execute()

    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".dcm"):
        return sitk.ReadImage(str(path))
    if suffix.endswith((".nii", ".nii.gz", ".mha", ".mhd", ".nrrd")):
        return sitk.ReadImage(str(path))

    # Fall back to letting SimpleITK infer the format from content.
    return sitk.ReadImage(str(path))


def hu_normalize(volume: np.ndarray, hu_min: float = HU_MIN, hu_max: float = HU_MAX) -> np.ndarray:
    """Clip Hounsfield Units to a lung window and rescale to [-1, 1]."""
    clipped = np.clip(volume, hu_min, hu_max)
    return (2.0 * (clipped - hu_min) / (hu_max - hu_min)) - 1.0


def denormalize_to_hu(volume: np.ndarray, hu_min: float = HU_MIN, hu_max: float = HU_MAX) -> np.ndarray:
    return ((volume + 1.0) / 2.0) * (hu_max - hu_min) + hu_min


def resample_to_shape(sitk_image: sitk.Image, target_shape: tuple[int, int, int]) -> sitk.Image:
    """Resample a volume to a fixed (D, H, W) voxel grid (isotropic
    working resolution), preserving physical extent."""
    original_size = sitk_image.GetSize()  # (x, y, z)
    original_spacing = sitk_image.GetSpacing()
    target_size = (target_shape[2], target_shape[1], target_shape[0])  # (x, y, z)

    new_spacing = [
        original_spacing[i] * (original_size[i] / target_size[i]) for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000.0)  # air/background in HU
    return resampler.Execute(sitk_image)


def volume_to_tensor(sitk_image: sitk.Image, working_size: int) -> torch.Tensor:
    """Full pipeline: resample -> HU clip/normalize -> torch tensor,
    shape (1, 1, D, H, W) in [-1, 1]."""
    resampled = resample_to_shape(sitk_image, (working_size, working_size, working_size))
    array = sitk.GetArrayFromImage(resampled).astype(np.float32)  # (z, y, x)
    normalized = hu_normalize(array)
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    return tensor


def tensor_to_nifti_bytes(tensor: torch.Tensor, reference: sitk.Image | None = None) -> bytes:
    """Convert a (1,1,D,H,W) tensor in [-1,1] back to HU space and encode
    as a .nii.gz byte stream for download."""
    array = tensor.squeeze(0).squeeze(0).cpu().numpy()
    hu_array = denormalize_to_hu(array).astype(np.float32)
    image = sitk.GetImageFromArray(hu_array)
    if reference is not None:
        image.SetSpacing(reference.GetSpacing())
        image.SetOrigin(reference.GetOrigin())
        image.SetDirection(reference.GetDirection())

    writer = sitk.ImageFileWriter()
    tmp_path = Path("/tmp") / f"_lungsynth_tmp_{id(tensor)}.nii.gz"
    writer.SetFileName(str(tmp_path))
    writer.Execute(image)
    data = tmp_path.read_bytes()
    tmp_path.unlink(missing_ok=True)
    return data


def mid_slice_preview_png(tensor: torch.Tensor, label: str = "") -> bytes:
    """Render the mid-axial slice of a (1,1,D,H,W) [-1,1] tensor as an
    8-bit PNG for the results grid / lightbox."""
    array = tensor.squeeze(0).squeeze(0).cpu().numpy()
    mid = array.shape[0] // 2
    sl = array[mid]
    sl_uint8 = (((sl + 1.0) / 2.0) * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(sl_uint8, mode="L").convert("RGB")

    if label:
        # Lightweight burned-in label so the PNG is self-describing even
        # if served standalone; the frontend also renders its own badge.
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        draw.rectangle([4, img.height - 20, 4 + 10 * len(label), img.height - 4], fill=(0, 0, 0))
        draw.text((8, img.height - 18), label, fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
