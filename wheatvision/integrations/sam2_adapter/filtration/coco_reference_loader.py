import json
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from wheatvision.core.types import ShapeDescriptor, CocoReferenceStatistics


class CocoEarReferenceLoader:
    """
    Loads a CVAT-exported COCO 1.0 JSON dataset of manually segmented wheat ears,
    rasterizes all ear polygons into binary masks, and computes per-mask shape
    descriptors. The aggregated descriptor statistics can be used as priors to
    guide SAM2 segmentation filtering.
    """

    def __init__(self, json_path: str) -> None:
        """
        Initializes the loader with the path to the COCO JSON file.

        Args:
            json_path (str): Path to the `instances_default.json` file inside the CVAT export.
        """
        self._json_path: str = json_path
        self._data: Optional[Dict[str, Any]] = None
        self._category_id_ear: Optional[int] = None

    def load(self) -> CocoReferenceStatistics:
        """
        Loads the COCO JSON file, decodes segmentations, and computes reference shape statistics.
        Returns a CocoReferenceStatistics dataclass.
        """
        with open(self._json_path, "r") as f:
            data = json.load(f)

        annotations = data.get("annotations", [])
        images = {img["id"]: img for img in data.get("images", [])}

        shape_descriptors: List[ShapeDescriptor] = []

        for ann in annotations:
            segmentation = ann.get("segmentation")
            image_info = images.get(ann["image_id"])
            if image_info is None or segmentation is None:
                continue

            mask = self._to_mask(
                segmentation, image_info["height"], image_info["width"]
            )
            if mask is None:
                continue

            descriptor = self._compute_shape_descriptor(mask)
            if descriptor.area > 0:
                shape_descriptors.append(descriptor)

        if not shape_descriptors:
            raise ValueError("No valid ear segmentations found in the COCO file.")

        areas = np.array(
            [shape_descriptor.area for shape_descriptor in shape_descriptors]
        )
        compactness = np.array(
            [shape_descriptor.compactness for shape_descriptor in shape_descriptors]
        )
        hu_moments = np.array(
            [shape_descriptor.hu_moments for shape_descriptor in shape_descriptors]
        )

        mean_hu = np.mean(hu_moments, axis=0)
        std_hu = np.std(hu_moments, axis=0)

        return CocoReferenceStatistics(
            mean_area=float(np.mean(areas)),
            std_area=float(np.std(areas)),
            mean_aspect_ratio=float(
                np.mean(
                    [
                        shape_descriptor.aspect_ratio
                        for shape_descriptor in shape_descriptors
                    ]
                )
            ),
            std_aspect_ratio=float(
                np.std(
                    [
                        shape_descriptor.aspect_ratio
                        for shape_descriptor in shape_descriptors
                    ]
                )
            ),
            mean_compactness=float(np.mean(compactness)),
            std_compactness=float(np.std(compactness)),
            mean_hu_moments=mean_hu,
            std_hu_moments=std_hu,
            all_descriptors=shape_descriptors,
        )

    def _to_mask(self, segmentation: Any, height: int, width: int) -> np.ndarray:
        """
        Converts a COCO segmentation entry to a binary mask.
        Handles both polygon and RLE encodings.
        """
        # Polygon segmentation (list of lists of coordinates)
        if isinstance(segmentation, list):
            mask = np.zeros((height, width), dtype=np.uint8)
            for polygon in segmentation:
                coords = np.array(polygon, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [coords], 255)
            return mask

        # RLE segmentation (dict with counts)
        if isinstance(segmentation, dict) and "counts" in segmentation:
            try:
                rle_mask = mask_utils.decode(segmentation)
                if rle_mask.ndim == 3:
                    rle_mask = np.any(rle_mask, axis=2)
                return (rle_mask.astype(np.uint8)) * 255
            except Exception as e:
                print(f"[COCO Loader] RLE decode failed: {e}")
                return None

        return None

    def _compute_shape_descriptor(self, binary_mask: np.ndarray) -> ShapeDescriptor:
        """
        Extracts geometric shape features from a binary mask.
        """
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return ShapeDescriptor(
                area=0,
                aspect_ratio=0,
                compactness=0,
                hu_moments=np.zeros(7, dtype=np.float32),
            )

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / float(h) if h > 0 else 0.0
        compactness = float(4 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0
        hu = cv2.HuMoments(cv2.moments(contour)).flatten()

        return ShapeDescriptor(
            area=area, aspect_ratio=aspect_ratio, compactness=compactness, hu_moments=hu
        )

    def _load_json(self, json_path: str) -> Dict[str, Any]:
        """Loads the COCO JSON content from disk."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"COCO JSON not found at: {json_path}")
        with open(json_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _resolve_ear_category_id(self, data: Dict[str, Any]) -> int:
        """Finds the category ID corresponding to the label 'ear' (case-insensitive)."""
        for category in data.get("categories", []):
            if category.get("name", "").lower() == "ear":
                return int(category["id"])
        raise ValueError("Category 'ear' not found in the COCO JSON file.")

    def _rasterize_annotation(
        self,
        annotation: Dict[str, Any],
        width: int,
        height: int,
    ) -> Optional[np.ndarray]:
        """
        Converts a polygonal segmentation into a binary mask.

        Args:
            annotation (Dict[str, Any]): COCO annotation dictionary with 'segmentation' key.
            width (int): Image width.
            height (int): Image height.

        Returns:
            Optional[np.ndarray]: Binary mask of shape (H, W), or None if invalid.
        """
        segmentation = annotation.get("segmentation", [])
        if not segmentation:
            return None

        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in segmentation:
            if not polygon or len(polygon) < 6:
                continue
            points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
            cv2.fillPoly(mask, [points.astype(np.int32)], 255)
        return mask

    def _compute_shape_descriptors(self, binary_mask: np.ndarray) -> Dict[str, Any]:
        """
        Computes key shape descriptors for a binary mask.

        Args:
            binary_mask (np.ndarray): Binary mask with foreground=255.

        Returns:
            Dict[str, Any]: Descriptor dictionary containing area, aspect_ratio, compactness, hu_moments.
        """
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return ShapeDescriptor(
                area=0.0,
                aspect_ratio=0.0,
                compactness=0.0,
                hu_moments=np.zeros(7, dtype=np.float32),
            )

        largest_contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest_contour))
        perimeter = float(cv2.arcLength(largest_contour, True))
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / float(h) if h > 0 else 0.0

        if perimeter > 0:
            compactness = float(4.0 * np.pi * area / (perimeter**2))
        else:
            compactness = 0.0

        moments = cv2.moments(largest_contour)
        hu_moments = cv2.HuMoments(moments).flatten()

        return ShapeDescriptor(
            area=area,
            aspect_ratio=aspect_ratio,
            compactness=compactness,
            hu_moments=hu_moments,
        )

    def _aggregate_statistics(
        self, descriptors: List[ShapeDescriptor]
    ) -> CocoReferenceStatistics:
        """
        Aggregates mean and standard deviation statistics from a list of ShapeDescriptor instances.
        """
        areas = np.array(
            [descriptor.area for descriptor in descriptors], dtype=np.float32
        )
        aspect_ratios = np.array(
            [d.aspect_ratio for d in descriptors], dtype=np.float32
        )
        compactnesses = np.array([d.compactness for d in descriptors], dtype=np.float32)
        hu_moments = np.stack([d.hu_moments for d in descriptors], axis=0)

        return CocoReferenceStatistics(
            mean_area=float(np.mean(areas)),
            mean_aspect_ratio=float(np.mean(aspect_ratios)),
            mean_compactness=float(np.mean(compactnesses)),
            mean_hu_moments=np.mean(hu_moments, axis=0),
            std_area=float(np.std(areas)),
            std_aspect_ratio=float(np.std(aspect_ratios)),
            std_compactness=float(np.std(compactnesses)),
            std_hu_moments=np.std(hu_moments, axis=0),
            all_descriptors=descriptors,
        )
