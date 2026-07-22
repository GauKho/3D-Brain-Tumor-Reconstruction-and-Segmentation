"""Convert nested region predictions to the BraTS label convention."""

import torch


def logits_to_brats_labels(
    logits: torch.Tensor,
    wt_threshold=0.5,
    tc_threshold=0.5,
    et_threshold=0.5,
) -> torch.Tensor:
    if logits.ndim != 5 or logits.shape[1] != 3:
        raise ValueError(f"Expected logits (B, 3, D, H, W), got {tuple(logits.shape)}")

    probabilities = torch.sigmoid(logits)
    wt = probabilities[:, 0] >= wt_threshold
    tc = probabilities[:, 1] >= tc_threshold
    et = probabilities[:, 2] >= et_threshold

    # Enforce the anatomical nesting ET ⊆ TC ⊆ WT.
    tc |= et
    wt |= tc

    labels = torch.zeros_like(wt, dtype=torch.int64)
    labels[wt] = 2
    labels[tc] = 1
    labels[et] = 4
    return labels
