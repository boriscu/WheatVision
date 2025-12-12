from typing import Callable, List, Optional, Tuple
import cv2
import numpy as np

from wheatvision.core.types import AspectRatioReferenceStats


DEFAULT_ASPECT_RATIO_REF = {
    "mean_ratio": 4.4358038902282715,
    "std_ratio": 1.9590742588043213,
    "count": 354,
}


class ShapeFilterService:
    """
    Filters segments by comparing each segment's bounding-box aspect ratio (height/width)
    to the reference mean ratio with a relative tolerance.
    """

    def __init__(self) -> None:
        pass

    def filter_segments(
        self,
        label_map: np.ndarray,
        reference_statistics: AspectRatioReferenceStats,
        ratio_tolerance: float = 0.30,  # ±30% around reference mean by default
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> np.ndarray:
        """
        Keep segments whose bounding-box aspect ratio is within
        [mean_ratio * (1 - tol), mean_ratio * (1 + tol)].

        Args:
            label_map: 2D integer array with background=0 and 1..K for segments.
            reference_statistics: AspectRatioReferenceStats (from COCO refs).
            ratio_tolerance: Fractional tolerance (e.g., 0.3 means ±30%).
            progress_callback: Optional progress reporter.

        Returns:
            2D int32 label map with background=0 and 1..N reindexed kept segments.
        """
        if label_map is None or label_map.ndim != 2:
            raise ValueError("label_map must be a 2D array with shape (H, W).")

        ref_mean: float = float(reference_statistics.mean_ratio)
        lower: float = ref_mean * (1.0 - float(ratio_tolerance))
        upper: float = ref_mean * (1.0 + float(ratio_tolerance))

        # Collect candidate segment ids (exclude 0)
        segment_ids = np.unique(label_map)
        segment_ids = segment_ids[segment_ids > 0]
        total = int(len(segment_ids))

        filtered_label_map = np.zeros_like(label_map, dtype=np.int32)
        next_id = 1

        for index, segment_id in enumerate(segment_ids, start=1):
            if progress_callback and index % 10 == 0:
                progress_callback(index / max(1, total), f"Filtering {index}/{total}…")

            mask_uint8 = (label_map == segment_id).astype(np.uint8) * 255
            bbox = self._compute_bounding_box(mask_uint8)
            if bbox is None:
                continue
            x, y, width, height = bbox
            if width <= 0:
                continue
            ratio = float(height) / float(width)

            if lower <= ratio <= upper:
                filtered_label_map[mask_uint8 > 0] = next_id
                next_id += 1

        # Reindex to 1..K (clean coloring)
        if filtered_label_map.max() > 0:
            remapped = np.zeros_like(filtered_label_map)
            unique_ids = np.unique(filtered_label_map)
            unique_ids = unique_ids[unique_ids > 0]
            for dense_id, src in enumerate(unique_ids, start=1):
                remapped[filtered_label_map == src] = dense_id
            filtered_label_map = remapped

        # Debug print (helps tune)
        kept = int(filtered_label_map.max())
        print(
            f"[RATIO FILTER] kept={kept}/{total} | "
            f"ref_mean={ref_mean:.3f} tol=±{ratio_tolerance:.2f} → "
            f"[{lower:.3f}, {upper:.3f}]"
        )
        if progress_callback:
            progress_callback(1.0, f"Filtering done. Kept {kept} segments.")
        return filtered_label_map

    def _compute_bounding_box(
        self, binary_mask: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Returns (x, y, width, height) for the largest external contour in the mask,
        or None if no contour exists.
        """
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(contour)
        return int(x), int(y), int(width), int(height)
