import cv2
from wheatvision.core.interfaces import (
    PreprocessingPipelineInterface,
    ForegroundMaskerInterface,
    SplitterInterface,
)

from wheatvision.core.types import ImageItem, PreprocessingResult, PreprocessingConfig


class PreprocessingPipeline(PreprocessingPipelineInterface):
    """Runs masking and cut finding to split ears and stalks."""

    def __init__(
        self,
        masker: ForegroundMaskerInterface,
        splitter: SplitterInterface,
        output_masked_halves: bool = False,
    ) -> None:
        """Compose pipeline from a masker and a splitter."""

        self.masker = masker
        self.splitter = splitter
        self._config = PreprocessingConfig()
        self._output_masked_halves = output_masked_halves

    def configure(self, config: PreprocessingConfig) -> None:
        """Apply configuration to self and children."""

        self._config = config
        self.masker.configure(config)
        self.splitter.configure(config)

    def run_on_item(self, item: ImageItem) -> PreprocessingResult:
        """Process a single image and return split results."""

        image = item.image_bgr

        foreground_mask = self.masker.make_foreground_mask(image)
        cut_position_y = self.splitter.find_cut(foreground_mask)
        density_profile, density_profile_smoothed = self.splitter.last_profiles

        masked_bgr = cv2.bitwise_and(image, image, mask=foreground_mask)
        base_image = masked_bgr if self._output_masked_halves else image

        ears_bgr = base_image[:cut_position_y, :].copy()
        stalks_bgr = base_image[cut_position_y:, :].copy()

        return PreprocessingResult(
            name=item.name,
            ears_bgr=ears_bgr,
            stalks_bgr=stalks_bgr,
            cut_position_y=int(cut_position_y),
            density_profile=density_profile,
            density_profile_smoothed=density_profile_smoothed,
            foreground_mask=foreground_mask,
        )
