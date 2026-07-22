"""Losses for nested BraTS regions."""

import torch
import torch.nn as nn


def brats_labels_to_regions(target: torch.Tensor) -> torch.Tensor:
    """Convert BraTS labels {0, 1, 2, 4} to channels [WT, TC, ET]."""
    return torch.stack(
        [target > 0, (target == 1) | (target == 4), target == 4], dim=1
    ).float()


class WeightedRegionDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5, et_weight_multiplier=1.0):
        super().__init__()
        weights = torch.tensor([4.51, 5.37, 6.12 * et_weight_multiplier], dtype=torch.float32)
        self.register_buffer("region_weights", weights / weights.sum())
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 5 or logits.shape[1] != 3:
            raise ValueError(f"Expected logits (B, 3, D, H, W), got {tuple(logits.shape)}")
        pred = torch.sigmoid(logits)
        target = brats_labels_to_regions(target_labels).to(device=logits.device, dtype=logits.dtype)
        dims = tuple(range(2, pred.ndim))
        intersection = (pred * target).sum(dim=dims)
        denominator = pred.sum(dim=dims) + target.sum(dim=dims)
        loss = 1.0 - (2.0 * intersection + self.epsilon) / (denominator + self.epsilon)
        return (self.region_weights.to(logits.dtype) * loss.mean(dim=0)).sum()
