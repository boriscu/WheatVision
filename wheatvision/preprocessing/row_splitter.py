import numpy as np

from wheatvision.core.interfaces import SplitterInterface
from wheatvision.core.types import PreprocessingConfig


class RowDensitySplitter(SplitterInterface):
    """Computes a vertical cut using a smoothed row-density valley."""

    def configure(self, config: PreprocessingConfig) -> None:
        """Cache split configuration parameters."""
        super().configure(config)

    def find_cut(self, foreground_mask: np.ndarray) -> int:
        """Return the cut position where ears transition to stalks."""
        import cv2
        import numpy as np
        from scipy.ndimage import gaussian_filter1d

        image_height, image_width = foreground_mask.shape[:2]

        # Select the tight horizontal span that contains plant pixels.
        plant_rows, plant_columns = np.where(foreground_mask > 0)
        if plant_columns.size > 0:
            x_min_index = max(0, int(plant_columns.min()) - 10)
            x_max_index = min(image_width, int(plant_columns.max()) + 10)
            plant_region_mask = foreground_mask[:, x_min_index:x_max_index]
        else:
            plant_region_mask = foreground_mask

        # Remove tall vertical structures (stalks) with a vertical opening.
        vertical_opening_length = max(
            15, int(self._config.split.vertical_opening_fraction * image_height)
        )
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, vertical_opening_length)
        )
        stalks_long_mask = cv2.morphologyEx(
            plant_region_mask, cv2.MORPH_OPEN, vertical_kernel
        )
        ears_mask = cv2.subtract(plant_region_mask, stalks_long_mask)

        # Compute the ear-bottom row as the 90th percentile of per-column bottoms.
        has_ear_pixel = ears_mask > 0
        valid_columns = has_ear_pixel.any(axis=0)
        cut_y = None
        if np.any(valid_columns):
            # Reverse rows so argmax finds first True from the bottom.
            bottom_offsets_from_bottom = np.argmax(has_ear_pixel[::-1, :], axis=0)
            column_bottom_y = image_height - 1 - bottom_offsets_from_bottom
            column_bottom_y = column_bottom_y[valid_columns]
            ear_bottom_y = int(np.percentile(column_bottom_y, 90))
            cut_y = ear_bottom_y + self._config.split.margin_pixels

        if cut_y is None:
            # Fallback: density valley on the plant region.
            density_profile = plant_region_mask.mean(axis=1) / 255.0
            smoothed = gaussian_filter1d(
                density_profile, sigma=self._config.split.gaussian_sigma
            )
            top_index = int(self._config.split.top_fraction * image_height)
            bottom_index = int(self._config.split.bottom_fraction * image_height)
            middle_band = smoothed[top_index:bottom_index]
            valley_local_index = int(np.argmin(middle_band))
            valley_y = top_index + valley_local_index
            cut_y = valley_y + self._config.split.margin_pixels

        min_cut_y = int(self._config.split.min_fraction * image_height)
        max_cut_y = int(self._config.split.max_fraction * image_height)
        cut_y = int(np.clip(cut_y, min_cut_y, max_cut_y))

        self._last_density = has_ear_pixel.mean(axis=1).astype(float)
        self._last_smoothed = self._last_density
        self._last_ears_mask = ears_mask

        return int(cut_y)

    @property
    def last_profiles(self) -> tuple:
        """Return raw and smoothed density profiles for inspection."""
        return getattr(self, "_last_density", None), getattr(
            self, "_last_smoothed", None
        )
