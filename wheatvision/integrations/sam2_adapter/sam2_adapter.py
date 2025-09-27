from dotenv import load_dotenv, find_dotenv
import numpy as np
from typing import Any, Optional
import torch

from wheatvision.core.types import Sam2Config

from wheatvision.integrations.sam2_adapter.exceptions import Sam2NotAvailable
from wheatvision.integrations.sam2_adapter.availability_checker import Sam2AvailabilityChecker
from wheatvision.integrations.sam2_adapter.config_resolver import Sam2ConfigResolver
from wheatvision.integrations.sam2_adapter.constructor import Sam2ConstructorsLoader
from wheatvision.integrations.sam2_adapter.predictor import Sam2PredictorBuilder
from wheatvision.integrations.sam2_adapter.oversegmentation_service import Sam2OversegmentationService

class Sam2Adapter:
    """
    Orchestrates SAM2 availability checking, configuration resolution, predictor construction, and oversegmentation.
    """

    def __init__(self) -> None:
        """
        Initializes the adapter, loads environment variables, and prepares helper services.
        """
                
        load_dotenv(find_dotenv(filename=".env", usecwd=True))
        self._configuration: Optional[Sam2Config] = None
        self._image_predictor = None
        self._availability_checker = Sam2AvailabilityChecker()
        self._is_available = self._availability_checker.try_import()
        self._config_resolver = Sam2ConfigResolver()
        self._overseg_service = Sam2OversegmentationService()

    def is_available(self) -> bool:
        """
        Reports whether the SAM2 package is importable in the current environment.

        Returns:
            bool: True if SAM2 is available, otherwise False.
        """
                
        return self._is_available

    def configure(self, configuration: Sam2Config) -> None:
        """
        Stores the provided configuration and clears any previously built predictor.

        Args:
            configuration (Sam2Config): SAM2 configuration including repo root, checkpoint path, model config path, device, and autocast flag.
        """

        self._configuration = configuration
        self._image_predictor = None

    def build(self) -> None:
        """
        Resolves configuration, loads SAM2 constructors, and builds the image predictor on the target device.

        Raises:
            Sam2NotAvailable: If SAM2 is not importable in the current environment.
            FileNotFoundError: If required config or checkpoint files cannot be located.
        """
                
        if not self._is_available:
            raise Sam2NotAvailable(
                "SAM2 is not importable. Ensure `pip install -e external/sam2_repo` succeeded."
            )
        if self._configuration is None:
            self._configuration = Sam2Config()

        resolved_configuration = self._config_resolver.resolve(self._configuration)
        constructors = Sam2ConstructorsLoader(self._is_available).load()
        self._image_predictor = Sam2PredictorBuilder().build(constructors, resolved_configuration)

    @torch.inference_mode()
    def oversegment(self, *args:Any, **kwargs:Any) -> np.ndarray:
        """
        Runs automatic oversegmentation using the underlying SAM2 predictor and returns an integer label map.

        Args:
            *args (Any): Positional arguments forwarded to Sam2OversegmentationService.oversegment.
            **kwargs (Any): Keyword arguments forwarded to Sam2OversegmentationService.oversegment.

        Returns:
            np.ndarray: Integer label map (H, W) with background=0 and positive integers as segment identifiers.

        Raises:
            Sam2NotAvailable: If SAM2 is unavailable and the predictor cannot be built.
        """
                
        if self._image_predictor is None:
            self.build()
        return self._overseg_service.oversegment(*args, image_predictor=self._image_predictor, **kwargs)

