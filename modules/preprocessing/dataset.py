"""PyTorch dataset around the wavelet preprocessing pipeline."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import PreprocessingConfig
from .pipeline import preprocess_case


class BraTSWaveletDataset(Dataset):
    def __init__(self, cases, augment=False, config=None):
        self.cases = list(cases)
        self.augment = augment
        self.config = config or PreprocessingConfig()

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        item = preprocess_case(self.cases[index], require_mask=True, config=self.config)
        image, mask = item["image"], item["mask"]
        if self.augment:
            for axis in range(3):
                if np.random.random() < 0.5:
                    image = np.flip(image, axis=axis + 1)
                    mask = np.flip(mask, axis=axis)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)).long(),
            "case_id": item["case_id"],
            "spacing": torch.tensor(item["spacing"], dtype=torch.float32),
        }
