from typing import Callable, List, Optional
import cv2
import numpy as np

from wheatvision.core.types import CocoReferenceStatistics, ShapeDescriptor


class ShapeFilterService:
    """
    Filters SAM2 segmentation results using shape statistics derived from
    reference ear masks. Each SAM2 segment is analyzed for geometric and
    invariant features and compared against the distribution of reference
    shapes to determine whether it resembles an ear.
    """

    def __init__(self) -> None:
        """Initializes the ShapeFilterService instance."""
        pass

    def filter_segments(
        self,
        label_map: np.ndarray,
        reference_statistics: CocoReferenceStatistics,
        size_tolerance: float = 0.5,
        compactness_tolerance: float = 0.2,
        hu_distance_threshold: float = 0.8,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> np.ndarray:
        """
        Filters the given SAM2 label map, keeping only ear-like segments based on
        geometric similarity to the provided reference statistics.

        Args:
            label_map (np.ndarray): Integer label map of shape (H, W) where 0 = background.
            reference_statistics (CocoReferenceStatistics): Reference shape statistics derived from manually segmented ears.
            size_tolerance (float): Acceptable relative deviation in area ratio (default = 0.5 → ±50%).
            compactness_tolerance (float): Maximum allowed absolute difference in compactness (default = 0.2).
            hu_distance_threshold (float): Maximum log-Hu distance allowed for shape similarity (default = 0.8).
            progress_callback (Optional[Callable[[float, str], None]]): Optional callback for progress reporting.

        Returns:
            np.ndarray: New label map of same shape as input, with non-ear-like segments removed (set to 0) and remaining segments reindexed 1..K.
        """
        if label_map is None or label_map.ndim != 2:
            raise ValueError("Input label_map must be a 2D integer array.")

        segment_ids = np.unique(label_map)
        segment_ids = segment_ids[segment_ids > 0]  # exclude background

        filtered_label_map = np.zeros_like(label_map, dtype=np.int32)
        kept_segments: List[int] = []
        new_id = 1

        total_segments = len(segment_ids)
        if total_segments == 0:
            return filtered_label_map

        for index, segment_id in enumerate(segment_ids, start=1):
            if progress_callback and index % 10 == 0:
                progress_callback(
                    index / total_segments,
                    f"Evaluating segment {index}/{total_segments}...",
                )

            segment_mask = (label_map == segment_id).astype(np.uint8) * 255
            descriptor = self._compute_shape_descriptor(segment_mask)

            if self._is_ear_like(
                descriptor=descriptor,
                reference=reference_statistics,
                size_tolerance=size_tolerance,
                compactness_tolerance=compactness_tolerance,
                hu_distance_threshold=hu_distance_threshold,
            ):
                filtered_label_map[segment_mask > 0] = new_id
                kept_segments.append(segment_id)
                new_id += 1

        if progress_callback:
            progress_callback(
                1.0,
                f"Filtered {len(kept_segments)}/{total_segments} segments retained.",
            )

        return filtered_label_map

    def _compute_shape_descriptor(self, binary_mask: np.ndarray) -> ShapeDescriptor:
        """
        Computes shape features for a single binary mask segment.

        Args:
            binary_mask (np.ndarray): Binary mask (255 = foreground).

        Returns:
            ShapeDescriptor: Shape features describing this segment.
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
        x, y, width, height = cv2.boundingRect(largest_contour)
        aspect_ratio = float(width) / float(height) if height > 0 else 0.0
        compactness = (
            float(4.0 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0
        )
        moments = cv2.moments(largest_contour)
        hu_moments = cv2.HuMoments(moments).flatten()

        return ShapeDescriptor(
            area=area,
            aspect_ratio=aspect_ratio,
            compactness=compactness,
            hu_moments=hu_moments,
        )

    def _is_ear_like(
        self,
        descriptor: ShapeDescriptor,
        reference: CocoReferenceStatistics,
        size_tolerance: float,
        compactness_tolerance: float,
        hu_distance_threshold: float,
    ) -> bool:
        """
        Determines if a segment's shape resembles the reference ear distribution.

        Args:
            descriptor (ShapeDescriptor): Shape descriptor of the current segment.
            reference (CocoReferenceStatistics): Aggregate reference statistics.
            size_tolerance (float): Acceptable relative deviation for area ratio.
            compactness_tolerance (float): Maximum allowed compactness deviation.
            hu_distance_threshold (float): Log-space Hu moment distance threshold.

        Returns:
            bool: True if the segment is considered ear-like.
        """
        if descriptor.area <= 0:
            return False

        # Area-based filtering
        area_ratio = descriptor.area / reference.mean_area
        if not (1.0 - size_tolerance <= area_ratio <= 1.0 + size_tolerance):
            return False

        # Compactness similarity
        compactness_difference = abs(
            descriptor.compactness - reference.mean_compactness
        )
        if compactness_difference > compactness_tolerance:
            return False

        # Hu-moment distance (log space, scale invariant)
        hu_distance = self._compute_log_hu_distance(
            descriptor.hu_moments, reference.mean_hu_moments
        )
        if hu_distance > hu_distance_threshold:
            return False

        return True

    def _compute_log_hu_distance(self, hu1: np.ndarray, hu2: np.ndarray) -> float:
        """
        Computes distance between two Hu moment vectors in log space.

        Args:
            hu1 (np.ndarray): First Hu moments vector.
            hu2 (np.ndarray): Second Hu moments vector.

        Returns:
            float: L2 distance in log-transformed space.
        """
        safe_hu1 = np.sign(hu1) * np.log10(np.abs(hu1) + 1e-12)
        safe_hu2 = np.sign(hu2) * np.log10(np.abs(hu2) + 1e-12)
        return float(np.linalg.norm(safe_hu1 - safe_hu2))
