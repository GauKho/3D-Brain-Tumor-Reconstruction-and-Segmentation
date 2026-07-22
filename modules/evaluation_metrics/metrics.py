"""Dice and HD95 metrics for BraTS WT, TC and ET regions."""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


def region_masks_from_labels(labels: np.ndarray):
    return {
        "WT": labels > 0,
        "TC": (labels == 1) | (labels == 4),
        "ET": labels == 4,
    }


def dice_binary(pred: np.ndarray, target: np.ndarray) -> float:
    pred, target = pred.astype(bool), target.astype(bool)
    pred_sum, target_sum = pred.sum(), target.sum()
    if pred_sum == 0 and target_sum == 0:
        return float("nan")
    if pred_sum == 0 or target_sum == 0:
        return 0.0
    return float(2.0 * np.logical_and(pred, target).sum() / (pred_sum + target_sum))


def surface_distances(mask_a: np.ndarray, mask_b: np.ndarray, spacing=(1.0, 1.0, 1.0)):
    structure = generate_binary_structure(3, 1)
    surface_a = mask_a ^ binary_erosion(mask_a, structure=structure, border_value=0)
    surface_b = mask_b ^ binary_erosion(mask_b, structure=structure, border_value=0)
    distance_to_b = distance_transform_edt(~surface_b, sampling=spacing)
    distance_to_a = distance_transform_edt(~surface_a, sampling=spacing)
    return np.concatenate([distance_to_b[surface_a], distance_to_a[surface_b]])


def hd95_binary(pred: np.ndarray, target: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    pred, target = pred.astype(bool), target.astype(bool)
    if pred.sum() == 0 and target.sum() == 0:
        return float("nan")
    if pred.sum() == 0 or target.sum() == 0:
        return float("inf")
    distances = surface_distances(pred, target, spacing)
    return float(np.percentile(distances, 95)) if distances.size else float("nan")


def safe_mean(values) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def compute_case_metrics(
    pred_labels: np.ndarray,
    target_labels: np.ndarray,
    spacing=(1.0, 1.0, 1.0),
    compute_hd95=False,
):
    pred_regions = region_masks_from_labels(pred_labels)
    target_regions = region_masks_from_labels(target_labels)
    metrics = {}
    for region in ("WT", "TC", "ET"):
        key = region.lower()
        metrics[f"dice_{key}"] = dice_binary(pred_regions[region], target_regions[region])
        if compute_hd95:
            metrics[f"hd95_{key}"] = hd95_binary(
                pred_regions[region], target_regions[region], spacing
            )
    return metrics
