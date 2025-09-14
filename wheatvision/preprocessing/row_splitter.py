import numpy as np
from scipy.ndimage import gaussian_filter1d

from wheatvision.core.interfaces import SplitterInterface
from wheatvision.core.types import PreprocessingConfig


class RowDensitySplitter(SplitterInterface):
    """Computes a vertical cut using a smoothed row-density valley."""

    def configure(self, config: PreprocessingConfig) -> None:
        """Cache split configuration parameters."""
        super().configure(config)

    def find_cut(self, foreground_mask: np.ndarray) -> int:
        """Return the cut position where ears transition to stalks."""
        height, width = foreground_mask.shape[:2]
        density_profile = foreground_mask.mean(axis=1) / 255.0
        sigma = self._config.split.gaussian_sigma
        smoothed = gaussian_filter1d(density_profile, sigma=sigma)

        top_idx = int(self._config.split.top_fraction * height)
        bottom_idx = int(self._config.split.bottom_fraction * height)
        middle = smoothed[top_idx:bottom_idx]
        valley_local = int(np.argmin(middle))
        valley_y = top_idx + valley_local

        min_y = int(self._config.split.min_fraction * height)
        max_y = int(self._config.split.max_fraction * height)
        valley_y = int(np.clip(valley_y, min_y, max_y))

        cut_y = min(height - 1, valley_y + self._config.split.margin_pixels)
        self._last_density = density_profile
        self._last_smoothed = smoothed
        return int(cut_y)

    @property
    def last_profiles(self) -> tuple:
        """Return raw and smoothed density profiles for inspection."""
        return getattr(self, "_last_density", None), getattr(
            self, "_last_smoothed", None
        )
