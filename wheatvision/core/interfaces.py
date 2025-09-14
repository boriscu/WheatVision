from abc import ABC, abstractmethod
from typing import Protocol
import numpy as np

from wheatvision.core.types import ImageItem, PreprocessingConfig, PreprocessingResult


class SupportsConfigure(Protocol):
    """Protocol for configurable components."""

    def configure(self, config: PreprocessingConfig) -> None:
        """Apply configuration to the component."""


class BaseComponent(ABC):
    """Base component with an optional configure hook."""

    def configure(self, config: PreprocessingConfig) -> None:
        """Receive configuration for later use."""
        self._config = config


class ForegroundMaskerInterface(BaseComponent):
    """Interface for foreground masking components."""

    @abstractmethod
    def make_foreground_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """Create a binary foreground mask from an image."""
        raise NotImplementedError


class SplitterInterface(BaseComponent):
    """Interface for computing the ears/stalks cut line."""

    @abstractmethod
    def find_cut(self, foreground_mask: np.ndarray) -> int:
        """Return the vertical cut position in pixels."""
        raise NotImplementedError


class PreprocessingPipelineInterface(BaseComponent):
    """Interface for a full preprocessing pipeline."""

    @abstractmethod
    def run_on_item(self, item: ImageItem) -> PreprocessingResult:
        """Run preprocessing on a single item."""
        raise NotImplementedError
