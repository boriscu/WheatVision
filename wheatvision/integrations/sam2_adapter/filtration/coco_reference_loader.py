import json
import os
from typing import Any, Dict, List, Optional

from wheatvision.core.types import AspectRatioReferenceStats


class CocoEarReferenceLoader:
    """
    Loads a COCO JSON and computes reference bounding-box aspect ratios (height / width)
    for the category named 'ear'. Uses annotation 'bbox' directly (no mask decoding).
    """

    def __init__(self, json_path: str) -> None:
        self._json_path: str = json_path
        self._data: Optional[Dict[str, Any]] = None
        self._ear_category_id: Optional[int] = None

    def load(self) -> AspectRatioReferenceStats:
        """Parse JSON, collect bbox aspect ratios for 'ear', and return stats."""
        self._data = self._load_json(self._json_path)
        self._ear_category_id = self._resolve_ear_category_id(self._data)

        ratios: List[float] = []
        for ann in self._data.get("annotations", []):
            if int(ann.get("category_id", -1)) != int(self._ear_category_id):
                continue
            bbox = ann.get("bbox", None)
            if not bbox or len(bbox) < 4:
                continue
            _, _, width, height = bbox  # COCO bbox = [x, y, w, h]
            width = float(width)
            height = float(height)
            if width <= 0.0:
                continue
            ratio = height / width
            if ratio > 0.0:
                ratios.append(ratio)

        if not ratios:
            raise ValueError("No valid 'ear' bboxes found in the COCO JSON.")

        # Simple mean/std
        import numpy as np

        ratios_np = np.asarray(ratios, dtype=np.float32)
        return AspectRatioReferenceStats(
            mean_ratio=float(ratios_np.mean()),
            std_ratio=float(ratios_np.std()),
            ratios=list(map(float, ratios)),
        )

    def _load_json(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"COCO JSON not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_ear_category_id(self, data: Dict[str, Any]) -> int:
        for cat in data.get("categories", []):
            if cat.get("name", "").lower() == "ear":
                return int(cat["id"])
        raise ValueError("Category 'ear' not found in the COCO JSON.")
