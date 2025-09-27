from pathlib import Path
import os
import torch

from wheatvision.core.types import Sam2Config

class Sam2ConfigResolver:
    """
    Resolves SAM2 configuration values from a Sam2Config object and environment variables into concrete paths and options.
    """
    def resolve(self, configuration: Sam2Config) -> dict:
        """
        Produces a resolved configuration dictionary containing repository path, checkpoint path, 
        Hydra model config name, and device.

        Args:
            configuration (Sam2Config): The user-provided configuration with optional overrides for repo root, 
            checkpoint, model config path, device, and autocast.

        Returns:
            dict: A mapping with keys:
                - "repository_path" (Path): Absolute path to the SAM2 repository.
                - "checkpoint_path" (Path): Absolute path to the model checkpoint file (.pt).
                - "model_config_name" (str): Hydra config name (e.g., "configs/sam2.1/sam2.1_hiera_s.yaml").
                - "device" (str): Target device string ("cuda" or "cpu").

        Raises:
            FileNotFoundError: If required environment variables are missing or if the checkpoint/config paths cannot be found.
        """
                
        repository_path = Path(
            configuration.sam2_repo_root or os.getenv("WHEATVISION_SAM2_REPO", "external/sam2_repo")
        ).resolve()

        checkpoint_path_string = configuration.checkpoint_path or os.getenv("WHEATVISION_SAM2_CKPT", "")
        if not checkpoint_path_string:
            raise FileNotFoundError("WHEATVISION_SAM2_CKPT is empty / missing in .env.")
        checkpoint_path = Path(checkpoint_path_string)
        if not checkpoint_path.is_absolute():
            checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            alternative_checkpoint_path = (repository_path / checkpoint_path_string).resolve()
            if alternative_checkpoint_path.is_file():
                checkpoint_path = alternative_checkpoint_path
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found at:\n  {checkpoint_path}\n"
                    f"(also tried repo-relative: {alternative_checkpoint_path})\n"
                    "Fix WHEATVISION_SAM2_CKPT in .env."
                )

        model_config_string = (configuration.model_cfg_path or os.getenv("WHEATVISION_SAM2_CFG", "")).strip()
        if not model_config_string:
            raise FileNotFoundError("WHEATVISION_SAM2_CFG is empty / missing in .env.")

        model_config_path = Path(model_config_string)
        if not model_config_path.is_absolute():
            model_config_path = (Path.cwd() / model_config_path).resolve()
        if not model_config_path.is_file():
            alternative_model_config_path = (repository_path / (model_config_string.lstrip("./"))).resolve()
            if alternative_model_config_path.is_file():
                model_config_path = alternative_model_config_path

        if model_config_path.is_file():
            try:
                model_config_posix = model_config_path.as_posix()
                sam2_configs_marker = "/sam2/configs/"
                marker_index = model_config_posix.rfind(sam2_configs_marker)
                if marker_index != -1:
                    model_config_name = "configs/" + model_config_posix[marker_index + len(sam2_configs_marker):]
                else:
                    generic_configs_marker = "/configs/"
                    generic_index = model_config_posix.rfind(generic_configs_marker)
                    if generic_index != -1:
                        model_config_name = "configs/" + model_config_posix[generic_index + len(generic_configs_marker):]
                    else:
                        model_config_name = model_config_path.name
            except Exception:
                model_config_name = model_config_path.name
        else:
            model_config_name = model_config_string

        device_name = (
            configuration.device
            or os.getenv("WHEATVISION_SAM2_DEVICE")
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        return {
            "repository_path": repository_path,
            "checkpoint_path": checkpoint_path,
            "model_config_name": model_config_name,
            "device": device_name,
        }
