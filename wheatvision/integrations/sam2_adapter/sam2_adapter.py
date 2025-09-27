from dotenv import load_dotenv, find_dotenv
import numpy as np
from typing import List, Optional, Tuple
import torch

from wheatvision.core.types import Sam2Config
from .exceptions import Sam2NotAvailable
from wheatvision.integrations.sam2_adapter.availability_checker import Sam2AvailabilityChecker
from wheatvision.integrations.sam2_adapter.config_resolver import Sam2ConfigResolver
from wheatvision.integrations.sam2_adapter.constructor import Sam2ConstructorsLoader
from wheatvision.integrations.sam2_adapter.predictor import Sam2PredictorBuilder
from wheatvision.integrations.sam2_adapter.oversegmentation_service import Sam2OversegmentationService

class Sam2Adapter:
    def __init__(self) -> None:
        load_dotenv(find_dotenv(filename=".env", usecwd=True))
        self._configuration: Optional[Sam2Config] = None
        self._image_predictor = None
        self._availability_checker = Sam2AvailabilityChecker()
        self._is_available = self._availability_checker.try_import()
        self._config_resolver = Sam2ConfigResolver()
        self._overseg_service = Sam2OversegmentationService()

    def is_available(self) -> bool:
        return self._is_available

    def configure(self, configuration: Sam2Config) -> None:
        self._configuration = configuration
        self._image_predictor = None

    def build(self) -> None:
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
    def oversegment(self, *args, **kwargs) -> np.ndarray:
        if self._image_predictor is None:
            self.build()
        return self._overseg_service.oversegment(*args, image_predictor=self._image_predictor, **kwargs)

