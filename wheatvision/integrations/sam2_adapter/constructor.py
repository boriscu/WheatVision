from wheatvision.integrations.sam2_adapter.exceptions import Sam2NotAvailable

class Sam2ConstructorsLoader:
    """
    Loads SAM2 constructor callables after verifying that the SAM2 package is available.
    """
        
    def __init__(self, is_available: bool) -> None:
        """
        Initialize the loader with a flag indicating whether the SAM2 package is importable.

        Args:
            is_available (bool): True if `import sam2` has succeeded in the current environment.
        """
        self._is_available = is_available

    def load(self):
        """
        Import and return SAM2's model builder and image predictor class.

        Returns:
            tuple: A pair `(build_sam2, SAM2ImagePredictor)` imported from the installed `sam2` package.

        Raises:
            Sam2NotAvailable: If the SAM2 package is not available in the environment.
        """
                
        if not self._is_available:
            raise Sam2NotAvailable("SAM2 import failed.")
        try:
            from external.sam2_repo.sam2.build_sam import build_sam2
            from external.sam2_repo.sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        return build_sam2, SAM2ImagePredictor
