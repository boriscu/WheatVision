import importlib
import numpy as np
import torch
import cv2

class Sam2OversegmentationService:
    @torch.inference_mode()
    def oversegment(
        self,
        image_bgr,
        image_predictor,
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
    ):

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
                image_predictor.model,
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
            key=lambda record: (record.get("area", 0), record.get("predicted_iou", 0.0)),
            reverse=True,
        )
        if maximum_segment_count and len(proposals) > maximum_segment_count:
            proposals = proposals[:maximum_segment_count]

        working_height, working_width = working_image_rgb.shape[:2]
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
                progress_callback(min(0.98, proposal_index / max(1, len(proposals))), f"Painting {proposal_index}/{len(proposals)}")

        if resize_scale != 1.0:
            label_map = cv2.resize(label_map, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        if progress_callback:
            progress_callback(1.0, f"Done (AMG). segments={int(label_map.max())}")
        print(f"[SAM2] AMG segments={int(label_map.max())}", flush=True)
        return label_map
