import importlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import numpy as np
import torch
import cv2


class Sam2OversegmentationService:
    """
    Provides automatic oversegmentation using SAM2's Automatic Mask Generator and returns a dense integer label map.
    """
        
    @torch.inference_mode()
    def oversegment(
        self,
        image_bgr: np.ndarray,
        image_predictor: Any,
        points_per_side: int = 48,
        predicted_intersection_over_union_threshold: float = 0.75,
        stability_score_threshold: float = 0.90,
        box_non_maximum_suppression_threshold: float = 0.7,
        crop_layer_count: int = 1,
        crop_overlap_ratio: float = 0.2,
        minimum_mask_region_area: int = 80,
        multi_mask_output: bool = True,
        points_per_batch: int = 64,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        downscale_long_side_pixels: Optional[int] = 1024,
        maximum_segment_count: int = 1500,
    ) -> np.ndarray:
        """
        Runs automatic mask generation on an image and composes results into an integer label map.

        Args:
            image_bgr (np.ndarray): Input image in BGR channel order.
            image_predictor (Any): Initialized SAM2 image predictor instance with a .model attribute.
            points_per_side (int): Grid density per crop side used by the generator.
            predicted_intersection_over_union_threshold (float): Quality filter using predicted IoU.
            stability_score_threshold (float): Quality filter using stability score.
            box_non_maximum_suppression_threshold (float): IoU threshold for NMS across masks.
            crop_layer_count (int): Number of crop layers to refine proposals.
            crop_overlap_ratio (float): Overlap ratio between crops.
            minimum_mask_region_area (int): Minimum area in pixels for accepting a mask.
            multi_mask_output (bool): Whether to generate multiple candidate masks per point.
            points_per_batch (int): Point prompts processed per batch by the model.
            progress_callback (Optional[Callable[[float, str], None]]): Progress reporter taking (fraction, message).
            downscale_long_side_pixels (Optional[int]): Downscale longer image side to this size for processing.
            maximum_segment_count (int): Maximum number of segments to keep after ranking.

        Returns:
            np.ndarray: Integer label map of shape (H, W) with background=0 and 1..K as segment ids.
        """

        (
            working_image_rgb,
            original_height,
            original_width,
            resize_scale,
        ) = self._prepare_working_image(image_bgr, downscale_long_side_pixels)

        amg_class = self._load_amg_class()

        proposals = self._run_amg(
            amg_class=amg_class,
            image_predictor=image_predictor,
            working_image_rgb=working_image_rgb,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            predicted_intersection_over_union_threshold=predicted_intersection_over_union_threshold,
            stability_score_threshold=stability_score_threshold,
            box_non_maximum_suppression_threshold=box_non_maximum_suppression_threshold,
            crop_layer_count=crop_layer_count,
            crop_overlap_ratio=crop_overlap_ratio,
            minimum_mask_region_area=minimum_mask_region_area,
            multi_mask_output=multi_mask_output,
            progress_callback=progress_callback,
        )

        proposals = self._maybe_relax_and_retry(
            proposals=proposals,
            amg_class=amg_class,
            image_predictor=image_predictor,
            working_image_rgb=working_image_rgb,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            crop_layer_count=crop_layer_count,
            crop_overlap_ratio=crop_overlap_ratio,
            minimum_mask_region_area=minimum_mask_region_area,
            progress_callback=progress_callback,
            box_non_maximum_suppression_threshold=box_non_maximum_suppression_threshold,
        )

        if not proposals:
            if progress_callback:
                progress_callback(1.0, "No segments found.")
            return np.zeros((original_height, original_width), dtype=np.int32)

        proposals = self._sort_and_cap_proposals(
            proposals=proposals,
            maximum_segment_count=maximum_segment_count,
        )

        label_map = self._paint_label_map(
            proposals=proposals,
            working_shape=working_image_rgb.shape[:2],
            minimum_mask_region_area=minimum_mask_region_area,
            progress_callback=progress_callback,
        )

        label_map = self._upsample_label_map(
            label_map=label_map,
            original_width=original_width,
            original_height=original_height,
            resize_scale=resize_scale,
        )

        if progress_callback:
            progress_callback(1.0, f"Done (AMG). segments={int(label_map.max())}")
        print(f"[SAM2] AMG segments={int(label_map.max())}", flush=True)
        return label_map

    def _prepare_working_image(
        self,
        image_bgr: np.ndarray,
        downscale_long_side_pixels: Optional[int],
    ) -> Tuple[np.ndarray, int, int, float]:
        """
        Converts BGR to RGB and optionally downsizes the image, returning processing metadata.

        Args:
            image_bgr (np.ndarray): Input image in BGR format.
            downscale_long_side_pixels (Optional[int]): Target size for the longer image side, if downscaling.

        Returns:
            Tuple[np.ndarray, int, int, float]: Working RGB image, original height, original width, and resize scale.
        """

        original_height, original_width = image_bgr.shape[:2]
        resize_scale = 1.0
        working_image_bgr = image_bgr
        if downscale_long_side_pixels and max(original_height, original_width) > downscale_long_side_pixels:
            resize_scale = downscale_long_side_pixels / float(max(original_height, original_width))
            working_image_bgr = cv2.resize(
                image_bgr,
                (int(round(original_width * resize_scale)), int(round(original_height * resize_scale))),
                interpolation=cv2.INTER_AREA,
            )
        working_image_rgb = working_image_bgr[..., ::-1].copy()
        return working_image_rgb, original_height, original_width, resize_scale

    def _load_amg_class(self):
        """
        Imports and returns the SAM2 Automatic Mask Generator class.

        Returns:
            Type[Any]: The `SAM2AutomaticMaskGenerator` class.
        """
                
        return importlib.import_module("sam2.automatic_mask_generator").SAM2AutomaticMaskGenerator

    def _run_amg(
        self,
        amg_class: Type[Any],
        image_predictor: Any,
        working_image_rgb: np.ndarray,
        points_per_side: int,
        points_per_batch: int,
        predicted_intersection_over_union_threshold: float,
        stability_score_threshold: float,
        box_non_maximum_suppression_threshold: float,
        crop_layer_count: int,
        crop_overlap_ratio: float,
        minimum_mask_region_area: int,
        multi_mask_output: bool,
        progress_callback: Optional[Callable[[float, str], None]],
    ) -> List[Dict[str, Any]]:
        """
        Runs SAM2 Automatic Mask Generator with the provided parameters on the working image.

        Args:
            amg_class (Type[Any]): The SAM2AutomaticMaskGenerator class.
            image_predictor (Any): Initialized SAM2 image predictor with a .model attribute.
            working_image_rgb (np.ndarray): RGB image used for mask generation.
            points_per_side (int): Grid density per crop side.
            points_per_batch (int): Number of points processed per batch.
            predicted_intersection_over_union_threshold (float): Predicted IoU threshold for filtering.
            stability_score_threshold (float): Stability score threshold for filtering.
            box_non_maximum_suppression_threshold (float): IoU threshold for NMS across masks.
            crop_layer_count (int): Number of crop layers.
            crop_overlap_ratio (float): Overlap ratio between crop windows.
            minimum_mask_region_area (int): Minimum mask area in pixels.
            multi_mask_output (bool): Whether to produce multiple candidate masks per point.
            progress_callback (Optional[Callable[[float, str], None]]): Optional progress reporter.

        Returns:
            List[Dict[str, Any]]: A list of mask proposal dictionaries produced by AMG.
        """

        automatic_mask_generator = amg_class(
            image_predictor.model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=predicted_intersection_over_union_threshold,
            stability_score_thresh=stability_score_threshold,
            mask_threshold=0.0,
            box_nms_thresh=box_non_maximum_suppression_threshold,
            crop_n_layers=crop_layer_count,
            crop_overlap_ratio=crop_overlap_ratio,
            min_mask_region_area=minimum_mask_region_area,
            output_mode="binary_mask",
            multimask_output=multi_mask_output,
        )
        if progress_callback:
            progress_callback(0.05, "Generating proposals (AMG)…")
        return automatic_mask_generator.generate(working_image_rgb)
    
    def _maybe_relax_and_retry(
        self,
        proposals: List[Dict[str, Any]],
        amg_class: Type[Any],
        image_predictor: Any,
        working_image_rgb: np.ndarray,
        points_per_side: int,
        points_per_batch: int,
        crop_layer_count: int,
        crop_overlap_ratio: float,
        minimum_mask_region_area: int,
        progress_callback: Optional[Callable[[float, str], None]],
        box_non_maximum_suppression_threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        Re-runs AMG with relaxed parameters if the initial proposal count is very low.

        Args:
            proposals (List[Dict[str, Any]]): Initial proposals produced by AMG.
            amg_class (Type[Any]): The SAM2AutomaticMaskGenerator class.
            image_predictor (Any): Initialized SAM2 image predictor with a .model attribute.
            working_image_rgb (np.ndarray): RGB image used for mask generation.
            points_per_side (int): Initial grid density used.
            points_per_batch (int): Point prompts processed per batch.
            crop_layer_count (int): Initial number of crop layers.
            crop_overlap_ratio (float): Initial crop overlap ratio.
            minimum_mask_region_area (int): Minimum mask area in pixels.
            progress_callback (Optional[Callable[[float, str], None]]): Optional progress reporter.
            box_non_maximum_suppression_threshold (float): IoU threshold for mask-level NMS.

        Returns:
            List[Dict[str, Any]]: Either the original proposals or a new set from the relaxed run.
        """

        if len(proposals) > 3:
            return proposals

        if progress_callback:
            progress_callback(0.10, "Few masks; retrying with denser/looser settings…")

        automatic_mask_generator = amg_class(
            image_predictor.model,
            points_per_side=max(64, points_per_side),
            points_per_batch=points_per_batch,
            pred_iou_thresh=0.70,
            stability_score_thresh=0.88,
            mask_threshold=0.0,
            box_nms_thresh=box_non_maximum_suppression_threshold,
            crop_n_layers=max(1, crop_layer_count),
            crop_overlap_ratio=max(0.2, crop_overlap_ratio),
            min_mask_region_area=max(20, minimum_mask_region_area // 2),
            output_mode="binary_mask",
            multimask_output=True,
        )
        return automatic_mask_generator.generate(working_image_rgb)

    def _sort_and_cap_proposals(
        self,
        proposals: List[Dict[str, Any]],
        maximum_segment_count: int,
    ) -> List[Dict[str, Any]]:
        """
        Sorts proposals by area and quality, and limits to a maximum count.

        Args:
            proposals (List[Dict[str, Any]]): Proposals to rank and filter.
            maximum_segment_count (int): Maximum number of proposals to keep.

        Returns:
            List[Dict[str, Any]]: The sorted and capped list of proposals.
        """

        proposals.sort(
            key=lambda record: (record.get("area", 0), record.get("predicted_iou", 0.0)),
            reverse=True,
        )
        if maximum_segment_count and len(proposals) > maximum_segment_count:
            proposals = proposals[:maximum_segment_count]
        return proposals

    def _paint_label_map(
        self,
        proposals: List[Dict[str, Any]],
        working_shape: Tuple[int, int],
        minimum_mask_region_area: int,
        progress_callback: Optional[Callable[[float, str], None]],
    ) -> np.ndarray:
        """
        Composes a dense integer label map from mask proposals.

        Args:
            proposals (List[Dict[str, Any]]): List of proposal dictionaries with a 'segmentation' mask.
            working_shape (Tuple[int, int]): Height and width of the working image.
            minimum_mask_region_area (int): Minimum area in pixels for accepting a mask region.
            progress_callback (Optional[Callable[[float, str], None]]): Optional progress reporter.

        Returns:
            np.ndarray: Label map aligned to the working image size with background=0.
        """

        working_height, working_width = working_shape
        label_map = np.zeros((working_height, working_width), dtype=np.int32)

        kept_count = 0
        for proposal_index, proposal in enumerate(proposals, 1):
            mask_array = proposal["segmentation"]
            mask_boolean = (mask_array > 0) if isinstance(mask_array, np.ndarray) else np.array(mask_array, dtype=bool)
            if mask_boolean.sum() < minimum_mask_region_area:
                continue
            kept_count += 1
            label_map[mask_boolean] = kept_count
            if progress_callback and (proposal_index % 25 == 0):
                progress_callback(
                    min(0.98, proposal_index / max(1, len(proposals))),
                    f"Painting {proposal_index}/{len(proposals)}",
                )
        return label_map

    def _upsample_label_map(
        self,
        label_map: np.ndarray,
        original_width: int,
        original_height: int,
        resize_scale: float,
    ) -> np.ndarray:
        """
        Resizes the label map back to the original image size using nearest-neighbor interpolation.

        Args:
            label_map (np.ndarray): Label map aligned to the working image size.
            original_width (int): Original image width.
            original_height (int): Original image height.
            resize_scale (float): Scale factor used when downscaling; 1.0 means no resizing is needed.

        Returns:
            np.ndarray: Label map resized to the original image dimensions.
        """
        
        if resize_scale != 1.0:
            label_map = cv2.resize(label_map, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        return label_map
