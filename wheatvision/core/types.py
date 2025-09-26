from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import numpy as np


class ScreenName(Enum):
    """UI screens enumeration."""

    PREPROCESSING = auto()
    SEGMENTATION = auto()


@dataclass
class HSVThresholds:
    """HSV thresholds for background masking."""

    hue_min: int = 0
    hue_max: int = 180
    saturation_max_background: int = 40
    value_min_background: int = 150


@dataclass
class MorphologyConfig:
    """Morphology kernel sizes and iterations."""

    open_kernel: int = 6
    open_iterations: int = 1
    close_kernel: int = 5
    close_iterations: int = 1


@dataclass
class SplitSearchConfig:
    """Configuration for valley search along image height."""

    top_fraction: float = 0.15
    bottom_fraction: float = 0.85
    gaussian_sigma: float = 8.0
    min_fraction: float = 0.25
    max_fraction: float = 0.75
    margin_pixels: int = 14
    vertical_opening_fraction: float = 0.08


@dataclass
class PreprocessingConfig:
    """Aggregate configuration for preprocessing."""

    hsv: HSVThresholds = HSVThresholds()
    morphology: MorphologyConfig = MorphologyConfig()
    split: SplitSearchConfig = SplitSearchConfig()


@dataclass
class ImageItem:
    """Container for an input image with metadata."""

    name: str
    image_bgr: np.ndarray


@dataclass
class PreprocessingResult:
    """Outputs for a single image preprocessing run."""

    name: str
    ears_bgr: np.ndarray
    stalks_bgr: np.ndarray
    cut_position_y: int
    density_profile: np.ndarray
    density_profile_smoothed: np.ndarray
    foreground_mask: np.ndarray


@dataclass
class Sam2Config:
    """
    Configuration for building a SAM2 predictor.

    You can leave these as None to use environment variables:
      - WHEATVISION_SAM2_REPO
      - WHEATVISION_SAM2_CKPT
      - WHEATVISION_SAM2_CFG
    """

    sam2_repo_root: Optional[str] = None
    checkpoint_path: Optional[str] = None  # .pt
    model_cfg_path: Optional[str] = None  # .yaml
    device: Optional[str] = None  # "cuda" | "cpu" | None=auto
    autocast: bool = True  # Use autocast on CUDA if available
