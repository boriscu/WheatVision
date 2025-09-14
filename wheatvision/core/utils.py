import numpy as np
import cv2


def ensure_three_channels(image: np.ndarray) -> np.ndarray:
    """Ensure image has three channels in BGR order."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image
