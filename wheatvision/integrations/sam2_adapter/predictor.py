from typing import Any, Callable, Dict, Tuple, Type


class Sam2PredictorBuilder:
    """
    Constructs a SAM2 image predictor from resolved configuration and SAM2 constructor callables.

    """

    def build(
        self,
        constructors: Tuple[Callable[[str, str], Any], Type[Any]],
        resolved_configuration: Dict[str, Any],
    ) -> Any:
        """
        Builds and returns a SAM2 image predictor on the requested device.

        Args:
            constructors (Tuple[Callable[[str, str], Any], Type[Any]]): A pair consisting of
                (build_sam2_function, SAM2ImagePredictorClass).
            resolved_configuration (Dict[str, Any]): Mapping with keys:
                - "model_config_name" (str): Hydra config name, e.g. "configs/sam2.1/sam2.1_hiera_s.yaml".
                - "checkpoint_path" (str | Path): Filesystem path to the checkpoint.
                - "device" (str): Target device string ("cuda" or "cpu").

        Returns:
            Any: An initialized SAM2 image predictor instance placed on the target device.
        """
        
        build_sam2_constructor, Sam2ImagePredictorClass = constructors
        sam2_model = build_sam2_constructor(
            str(resolved_configuration["model_config_name"]),
            str(resolved_configuration["checkpoint_path"]),
        )
        image_predictor = Sam2ImagePredictorClass(sam2_model)
        device_name = resolved_configuration["device"]
        image_predictor.model.to(device_name)
        return image_predictor
