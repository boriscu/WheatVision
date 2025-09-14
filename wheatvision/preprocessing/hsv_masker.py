import numpy as np
import cv2

from wheatvision.core.interfaces import ForegroundMaskerInterface
from wheatvision.core.types import PreprocessingConfig


class HSVForegroundMasker(ForegroundMaskerInterface):
    """Foreground masker using HSV thresholds and morphology."""

    def configure(self, config: PreprocessingConfig) -> None:
        """Store configuration for masking and morphology."""

        super().configure(config)

    def make_foreground_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """Create a clean foreground mask separating plant from white background."""

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        thresholds = self._config.hsv
        background = cv2.inRange(
            hsv,
            (thresholds.hue_min, 0, thresholds.value_min_background),
            (thresholds.hue_max, thresholds.saturation_max_background, 255),
        )
        foreground = cv2.bitwise_not(background)

        morph = self._config.morphology
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph.open_kernel, morph.open_kernel)
        )
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph.close_kernel, morph.close_kernel)
        )

        foreground = cv2.morphologyEx(
            foreground, cv2.MORPH_OPEN, open_kernel, iterations=morph.open_iterations
        )
        foreground = cv2.morphologyEx(
            foreground, cv2.MORPH_CLOSE, close_kernel, iterations=morph.close_iterations
        )

        return foreground
