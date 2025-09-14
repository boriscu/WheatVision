import os
import cv2
import numpy as np
from typing import List

from wheatvision.core.types import ImageItem


class FileManager:
    """Handles reading and writing image files."""

    def read_images(self, file_paths: List[str]) -> List[ImageItem]:
        """Load images from disk into ImageItem structures."""

        items: List[ImageItem] = []
        for path in file_paths:
            image_bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if image_bgr is None:
                continue
            name = os.path.basename(path)
            items.append(ImageItem(name=name, image_bgr=image_bgr))
        return items

    def write_image(self, output_folder: str, name: str, image_bgr: np.ndarray) -> str:
        """Write an image to output folder and return its path."""

        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, name)
        cv2.imwrite(out_path, image_bgr)
        return out_path
