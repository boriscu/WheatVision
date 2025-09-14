from dataclasses import dataclass
from enum import Enum, auto
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
    value_min_background: int = 200


@dataclass
class MorphologyConfig:
    """Morphology kernel sizes and iterations."""

    open_kernel: int = 3
    open_iterations: int = 1
    close_kernel: int = 5
    close_iterations: int = 1


@dataclass
class SplitSearchConfig:
    """Configuration for valley search along image height."""

    top_fraction: float = 0.2
    bottom_fraction: float = 0.8
    gaussian_sigma: float = 5.0
    min_fraction: float = 0.25
    max_fraction: float = 0.75
    margin_pixels: int = 10


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
