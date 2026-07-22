"""Configuration for the notebook's 9-channel preprocessing pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class PreprocessingConfig:
    modalities: Tuple[str, ...] = ("t1ce", "t2", "flair")
    patch_size: Tuple[int, int, int] = (128, 128, 128)
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    normalization: str = "minmax"
    percentile_clip: bool = True
    percentile_range: Tuple[float, float] = (1.0, 99.0)
    wavelet_name: str = "sym8"
    wavelet_level: int = 2
    wavelet_mode: str = "symmetric"
    cache_dir: Optional[Path] = None

    @property
    def input_channels(self) -> int:
        return len(self.modalities) + 3 * self.wavelet_level
