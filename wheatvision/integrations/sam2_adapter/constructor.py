from wheatvision.integrations.sam2_adapter.exceptions import Sam2NotAvailable

class Sam2ConstructorsLoader:
    def __init__(self, is_available: bool) -> None:
        self._is_available = is_available

    def load(self):
        if not self._is_available:
            raise Sam2NotAvailable("SAM2 import failed.")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        return build_sam2, SAM2ImagePredictor
