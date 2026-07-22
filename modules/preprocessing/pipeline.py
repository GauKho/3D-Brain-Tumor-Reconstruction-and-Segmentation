"""End-to-end conversion of one BraTS case to nine model channels."""

import numpy as np

from .config import PreprocessingConfig
from .transforms import (
    center_crop_pad_3d,
    check_same_geometry,
    load_nifti,
    normalize_in_brain,
    resample_volume,
    wavelet_energy_maps,
)


def preprocess_case(case, require_mask=True, config=None):
    config = config or PreprocessingConfig()
    arrays, images = {}, {}
    for modality in config.modalities:
        arrays[modality], images[modality] = load_nifti(case[modality])
    if require_mask:
        segmentation, images["seg"] = load_nifti(case["seg"])
    else:
        segmentation = None
    check_same_geometry(images, case["case_id"])

    reference = images[config.modalities[0]]
    spacing = tuple(float(x) for x in reference.header.get_zooms()[:3])
    cropped = {}
    for modality, array in arrays.items():
        array = resample_volume(array, spacing, config.target_spacing, order=1)
        cropped[modality] = center_crop_pad_3d(array, config.patch_size)
    if segmentation is not None:
        segmentation = resample_volume(segmentation, spacing, config.target_spacing, order=0)
        segmentation = center_crop_pad_3d(segmentation, config.patch_size).astype(np.int64)
        segmentation[~np.isin(segmentation, [0, 1, 2, 4])] = 0

    brain_mask = np.logical_or.reduce([array > 0 for array in cropped.values()])
    normalized = {name: normalize_in_brain(array, brain_mask, config) for name, array in cropped.items()}
    t1ce, t2, flair = (normalized[name] for name in ("t1ce", "t2", "flair"))
    wavelet_channels = []
    for source in (t1ce, t1ce - flair, t2 - t1ce):
        wavelet_channels.extend(wavelet_energy_maps(source, brain_mask, config))
    image = np.stack([t1ce, t2, flair, *wavelet_channels]).astype(np.float32)
    image[:, ~brain_mask] = 0
    if image.shape != (config.input_channels, *config.patch_size):
        raise RuntimeError(f"{case['case_id']}: invalid image shape {image.shape}")
    return {
        "case_id": case["case_id"], "image": image, "mask": segmentation,
        "flair": flair, "brain_mask": brain_mask, "affine": reference.affine,
        "spacing": spacing, "original_shape": reference.shape[:3],
    }
