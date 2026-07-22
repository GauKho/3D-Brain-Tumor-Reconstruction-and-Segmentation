"""Geometry, normalization and wavelet transforms."""

import nibabel as nib
import numpy as np
import pywt
from scipy.ndimage import zoom
from skimage.transform import resize

from .config import PreprocessingConfig


def load_nifti(path, dtype=np.float32):
    image = nib.load(str(path))
    array = image.get_fdata(dtype=np.float32).astype(dtype, copy=False)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0), image


def check_same_geometry(images, case_id):
    shapes = {name: image.shape[:3] for name, image in images.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"{case_id}: modality shape mismatch: {shapes}")


def resample_volume(volume, spacing, target_spacing, order):
    spacing, target = np.asarray(spacing), np.asarray(target_spacing)
    if np.allclose(spacing, target, atol=1e-3):
        return volume
    factors = np.round(np.asarray(volume.shape) * spacing / target).astype(int) / volume.shape
    return zoom(volume, factors, order=order).astype(volume.dtype, copy=False)


def center_crop_pad_3d(volume, target_shape, pad_value=0, return_meta=False):
    output = np.full(target_shape, pad_value, dtype=volume.dtype)
    source_slices, target_slices = [], []
    for source_len, target_len in zip(volume.shape, target_shape):
        source_start = max(0, (source_len - target_len) // 2)
        target_start = max(0, (target_len - source_len) // 2)
        length = min(source_len, target_len)
        source_slices.append(slice(source_start, source_start + length))
        target_slices.append(slice(target_start, target_start + length))
    source_slices, target_slices = tuple(source_slices), tuple(target_slices)
    output[target_slices] = volume[source_slices]
    if return_meta:
        return output, {"original_shape": volume.shape, "src_slices": source_slices, "dst_slices": target_slices}
    return output


def normalize_in_brain(volume, brain_mask, config: PreprocessingConfig):
    output = np.zeros_like(volume, dtype=np.float32)
    values = volume[brain_mask]
    if values.size == 0:
        return output
    if config.percentile_clip:
        low, high = np.percentile(values, config.percentile_range)
        volume = np.clip(volume, low, high)
        values = volume[brain_mask]
    if config.normalization == "zscore":
        scale, offset = float(values.std()), float(values.mean())
    elif config.normalization == "minmax":
        scale, offset = float(values.max() - values.min()), float(values.min())
    else:
        raise ValueError(f"Unknown normalization: {config.normalization}")
    if scale >= 1e-6:
        output[brain_mask] = (volume[brain_mask] - offset) / scale
    return output


def wavelet_energy_maps(source, brain_mask, config: PreprocessingConfig):
    coefficients = pywt.wavedecn(
        source.astype(np.float32), config.wavelet_name, mode=config.wavelet_mode,
        level=config.wavelet_level, axes=(0, 1, 2)
    )
    maps = []
    for detail in reversed(coefficients[1:]):
        energy = np.sqrt(sum(np.square(np.abs(x).astype(np.float32)) for x in detail.values()) + 1e-8)
        energy = resize(energy, config.patch_size, order=1, mode="reflect", anti_aliasing=False, preserve_range=True)
        maps.append(normalize_in_brain(energy, brain_mask, config))
    return maps
