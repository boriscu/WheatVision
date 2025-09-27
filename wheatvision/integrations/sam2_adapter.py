from contextlib import contextmanager
from dotenv import load_dotenv, find_dotenv
from typing import List, Optional, Tuple
import importlib
import numpy as np
import torch

from wheatvision.core.types import Sam2Config


class Sam2NotAvailable(RuntimeError):
    """Raised when SAM2 is not importable/installed."""


class Sam2Adapter:
    """
    Thin OOP wrapper around SAM2's image predictor.
    """

    def __init__(self) -> None:
        load_dotenv(find_dotenv(filename=".env", usecwd=True))
        self._configuration: Optional[Sam2Config] = None
        self._image_predictor = None
        self._is_available = self._try_import_sam2()

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

        resolved_configuration = self._resolve_configuration(self._configuration)
        build_sam2_constructor, Sam2ImagePredictorClass = self._get_sam2_constructors()

        sam2_model = build_sam2_constructor(
            str(resolved_configuration["model_config_name"]),
            str(resolved_configuration["checkpoint_path"]),
        )
        image_predictor = Sam2ImagePredictorClass(sam2_model)

        device_name = resolved_configuration["device"]
        image_predictor.model.to(device_name)
        self._image_predictor = image_predictor

    @torch.inference_mode()
    def oversegment(
        self,
        image_bgr: np.ndarray,
        *,
        points_per_side: int = 48,
        predicted_intersection_over_union_threshold: float = 0.75,
        stability_score_threshold: float = 0.90,
        box_non_maximum_suppression_threshold: float = 0.7,
        crop_layer_count: int = 1,
        crop_overlap_ratio: float = 0.2,
        minimum_mask_region_area: int = 80,
        multi_mask_output: bool = True,
        points_per_batch: int = 64,
        progress_callback=None,
        downscale_long_side_pixels: int | None = 1024,
        maximum_segment_count: int = 1500,
    ) -> np.ndarray:
        import cv2

        if self._image_predictor is None:
            self.build()

        original_height, original_width = image_bgr.shape[:2]
        resize_scale = 1.0
        working_image_bgr = image_bgr
        if (
            downscale_long_side_pixels
            and max(original_height, original_width) > downscale_long_side_pixels
        ):
            resize_scale = downscale_long_side_pixels / float(
                max(original_height, original_width)
            )
            working_image_bgr = cv2.resize(
                image_bgr,
                (int(round(original_width * resize_scale)), int(round(original_height * resize_scale))),
                interpolation=cv2.INTER_AREA,
            )
        working_image_rgb = working_image_bgr[..., ::-1].copy()

        AutomaticMaskGeneratorClass = importlib.import_module(
            "sam2.automatic_mask_generator"
        ).SAM2AutomaticMaskGenerator

        def run_automatic_mask_generator(
            points_per_side_local: int,
            predicted_iou_threshold_local: float,
            stability_score_threshold_local: float,
            crop_layer_count_local: int,
            crop_overlap_ratio_local: float,
            minimum_mask_region_area_local: int,
            multi_mask_output_local: bool,
        ):
            automatic_mask_generator = AutomaticMaskGeneratorClass(
                self._image_predictor.model,
                points_per_side=points_per_side_local,
                points_per_batch=points_per_batch,
                pred_iou_thresh=predicted_iou_threshold_local,
                stability_score_thresh=stability_score_threshold_local,
                mask_threshold=0.0,
                box_nms_thresh=box_non_maximum_suppression_threshold,
                crop_n_layers=crop_layer_count_local,
                crop_overlap_ratio=crop_overlap_ratio_local,
                min_mask_region_area=minimum_mask_region_area_local,
                output_mode="binary_mask",
                multimask_output=multi_mask_output_local,
            )
            if progress_callback:
                progress_callback(0.05, "Generating proposals (AMG)…")
            return automatic_mask_generator.generate(working_image_rgb)

        proposals = run_automatic_mask_generator(
            points_per_side,
            predicted_intersection_over_union_threshold,
            stability_score_threshold,
            crop_layer_count,
            crop_overlap_ratio,
            minimum_mask_region_area,
            multi_mask_output,
        )

        if len(proposals) <= 3:
            if progress_callback:
                progress_callback(0.10, "Few masks; retrying with denser/looser settings…")
            proposals = run_automatic_mask_generator(
                points_per_side_local=max(64, points_per_side),
                predicted_iou_threshold_local=0.70,
                stability_score_threshold_local=0.88,
                crop_layer_count_local=max(1, crop_layer_count),
                crop_overlap_ratio_local=max(0.2, crop_overlap_ratio),
                minimum_mask_region_area_local=max(20, minimum_mask_region_area // 2),
                multi_mask_output_local=True,
            )

        if not proposals:
            if progress_callback:
                progress_callback(1.0, "No segments found.")
            return np.zeros((original_height, original_width), dtype=np.int32)

        proposals.sort(
            key=lambda record: (
                record.get("area", 0),
                record.get("predicted_iou", 0.0),
            ),
            reverse=True,
        )
        if maximum_segment_count and len(proposals) > maximum_segment_count:
            proposals = proposals[:maximum_segment_count]

        working_height, working_width = working_image_rgb.shape[:2]
        label_map = np.zeros((working_height, working_width), dtype=np.int32)

        kept_count = 0
        for proposal_index, proposal in enumerate(proposals, 1):
            mask_array = proposal["segmentation"]
            mask_boolean = (
                (mask_array > 0)
                if isinstance(mask_array, np.ndarray)
                else np.array(mask_array, dtype=bool)
            )
            if mask_boolean.sum() < minimum_mask_region_area:
                continue
            kept_count += 1
            label_map[mask_boolean] = kept_count
            if progress_callback and (proposal_index % 25 == 0):
                progress_callback(
                    min(0.98, proposal_index / max(1, len(proposals))),
                    f"Painting {proposal_index}/{len(proposals)}",
                )

        if resize_scale != 1.0:
            label_map = cv2.resize(
                label_map, (original_width, original_height), interpolation=cv2.INTER_NEAREST
            )

        if progress_callback:
            progress_callback(1.0, f"Done (AMG). segments={int(label_map.max())}")
        print(f"[SAM2] AMG segments={int(label_map.max())}", flush=True)
        return label_map

    @torch.inference_mode()
    def predict_from_points(
        self,
        image_bgr: np.ndarray,
        foreground_points: List[Tuple[int, int]] | None = None,
        background_points: List[Tuple[int, int]] | None = None,
    ) -> np.ndarray:
        if self._image_predictor is None:
            self.build()

        image_rgb = image_bgr[..., ::-1].copy()
        self._image_predictor.set_image(image_rgb)

        point_coordinates, point_labels = self._pack_points(foreground_points, background_points)

        use_autocast_flag = bool(self._configuration.autocast if self._configuration else True)
        use_cuda_flag = self._image_predictor.model.device.type == "cuda"
        autocast_context = self._autocast_context(enabled_flag=(use_autocast_flag and use_cuda_flag))

        with autocast_context:
            masks, scores, _ = self._image_predictor.predict(
                point_coords=point_coordinates,
                point_labels=point_labels,
                multimask_output=False,
            )

        binary_mask = masks[0].astype(np.uint8)
        return binary_mask

    def _try_import_sam2(self) -> bool:
        try:
            import sam2  # noqa: F401
            return True
        except Exception:
            return False

    def _get_sam2_constructors(self):
        if not self._is_available:
            raise Sam2NotAvailable("SAM2 import failed.")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        return build_sam2, SAM2ImagePredictor

    def _resolve_configuration(self, configuration: Sam2Config) -> dict:
        from pathlib import Path
        import os

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
                    model_config_name = "configs/" + model_config_posix[marker_index + len(sam2_configs_marker) :]
                else:
                    generic_configs_marker = "/configs/"
                    generic_index = model_config_posix.rfind(generic_configs_marker)
                    if generic_index != -1:
                        model_config_name = "configs/" + model_config_posix[generic_index + len(generic_configs_marker) :]
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

    @staticmethod
    def _pack_points(
        foreground_points: List[Tuple[int, int]] | None,
        background_points: List[Tuple[int, int]] | None,
    ):
        foreground_points = foreground_points or []
        background_points = background_points or []
        if not foreground_points and not background_points:
            return None, None

        points_array = np.array(foreground_points + background_points, dtype=np.int32)
        labels_array = np.array(
            [1] * len(foreground_points) + [0] * len(background_points), dtype=np.int32
        )
        return points_array, labels_array

    @contextmanager
    def _autocast_context(self, enabled_flag: bool):
        if enabled_flag:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                yield
        else:
            yield
