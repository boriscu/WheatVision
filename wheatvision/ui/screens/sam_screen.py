from typing import List, Tuple
import os
import cv2
import gradio as gr
import numpy as np
import tempfile
import zipfile
import torch

from wheatvision.core.types import AspectRatioReferenceStats
from wheatvision.integrations.sam2_adapter import Sam2Adapter, Sam2NotAvailable

from wheatvision.integrations.sam2_adapter.filtration.coco_reference_loader import (
    CocoEarReferenceLoader,
)
from wheatvision.integrations.sam2_adapter.filtration.shape_filter_service import (
    DEFAULT_ASPECT_RATIO_REF,
    ShapeFilterService,
)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _colorize_labels(label_map: np.ndarray) -> np.ndarray:
    height, width = label_map.shape
    vis = np.zeros((height, width, 3), dtype=np.uint8)
    if label_map.max() == 0:
        return vis
    rng = np.random.default_rng(42)
    colors = rng.integers(50, 230, size=(label_map.max() + 1, 3), dtype=np.uint8)
    colors[0] = np.array([0, 0, 0], dtype=np.uint8)
    return colors[label_map]


def _save_uint16_png(path: str, arr: np.ndarray) -> None:
    arr16 = np.ascontiguousarray(arr.astype(np.uint16))
    cv2.imwrite(path, arr16)


def _scaled_gray_for_display(label_map: np.ndarray) -> np.ndarray:
    max_id = int(label_map.max())
    if max_id <= 0:
        disp = np.zeros_like(label_map, dtype=np.uint8)
    else:
        scale = max(1, 255 // max_id)
        disp = (label_map * scale).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)


# -------------------------------------------------------------------------
# SAM2 Processing
# -------------------------------------------------------------------------


def _process_sam_batch(
    files: List[str],
    *,
    points_per_side: int,
    min_mask_region_area: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    crop_n_layers: int,
    crop_overlap_ratio: float,
    downscale_long_side: int,
    max_segments: int,
    multimask_output: bool,
    maximum_mask_region_area: int,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], str]:
    """
    Run SAM2 oversegmentation over a list of filepaths.
    Returns (colorized_previews, label_previews_grayRGB, label_maps, zip_path).
    """

    adapter = Sam2Adapter()
    if not adapter.is_available():
        reason = getattr(adapter, "availability_error", lambda: None)() or "unknown"
        raise Sam2NotAvailable(f"SAM2 not available: {reason}")

    adapter.build()

    color_previews: List[np.ndarray] = []
    gray_previews: List[np.ndarray] = []
    label_maps: List[np.ndarray] = []

    tmpdir = tempfile.mkdtemp(prefix="wheatvision_sam2_")
    zip_path = os.path.join(tmpdir, "sam2_overseg_outputs.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        total = len(files or [])
        for idx, path in enumerate(files or []):
            progress((idx, total), desc=f"Segmenting {os.path.basename(path)}")

            image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue

            label_map = adapter.oversegment(
                image_bgr=_to_uint8(image_bgr),
                points_per_side=int(points_per_side),
                predicted_intersection_over_union_threshold=float(pred_iou_thresh),
                stability_score_threshold=float(stability_score_thresh),
                box_non_maximum_suppression_threshold=0.7,
                crop_layer_count=int(crop_n_layers),
                crop_overlap_ratio=float(crop_overlap_ratio),
                minimum_mask_region_area=int(min_mask_region_area),
                maximum_mask_region_area=int(maximum_mask_region_area),
                multi_mask_output=bool(multimask_output),
                points_per_batch=64,
                progress_callback=progress,
                downscale_long_side_pixels=int(downscale_long_side),
                maximum_segment_count=int(max_segments),
            )

            vis_rgb = _colorize_labels(label_map)
            disp_rgb = _scaled_gray_for_display(label_map)

            color_previews.append(vis_rgb)
            gray_previews.append(disp_rgb)
            label_maps.append(label_map)

            base = os.path.splitext(os.path.basename(path))[0]
            lbl_name = f"{base}_labels_uint16.png"
            vis_name = f"{base}_labels_color.png"

            lbl_path = os.path.join(tmpdir, lbl_name)
            vis_path = os.path.join(tmpdir, vis_name)

            _save_uint16_png(lbl_path, label_map)
            cv2.imwrite(vis_path, cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))

            zipf.write(lbl_path, lbl_name)
            zipf.write(vis_path, vis_name)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return color_previews, gray_previews, label_maps, zip_path


# -------------------------------------------------------------------------
# Gradio UI
# -------------------------------------------------------------------------


def build_sam_tab():
    """Batch-only SAM2 oversegmentation UI (Automatic Mask Generator + optional shape filtering)."""

    gr.Markdown(
        "#### SAM2 Oversegmentation – Batch (Automatic Mask Generator + Shape Filtering)"
    )

    adapter = Sam2Adapter()
    if not adapter.is_available():
        reason = getattr(adapter, "availability_error", lambda: None)() or "unknown"
        gr.Markdown(
            f"> ⚠️ **SAM2 isn’t installed or not importable.** ({reason})  \n"
            "Install SAM2 and configure your `.env`."
        )
        return

    with gr.Row():
        with gr.Column(scale=1):
            files = gr.Files(
                label="Batch images (JPG/PNG)", file_types=["image"], type="filepath"
            )
            ref_json = gr.File(
                label="Optional COCO Reference (defaults to internal wheat-ear ratio stats if omitted)"
            )

            with gr.Accordion("Mask Generator Parameters", open=False):
                points_per_side = gr.Slider(
                    8, 64, step=1, value=32, label="points_per_side"
                )
                min_area = gr.Slider(
                    0, 2000, step=50, value=150, label="min_mask_region_area (px)"
                )
                pred_iou = gr.Slider(
                    0.70, 0.99, step=0.01, value=0.88, label="pred_iou_thresh"
                )
                stab = gr.Slider(
                    0.80, 0.99, step=0.01, value=0.95, label="stability_score_thresh"
                )
                crop_layers = gr.Dropdown(
                    choices=[0, 1], value=0, label="crop_n_layers"
                )
                crop_overlap = gr.Slider(
                    0.0, 0.6, step=0.05, value=0.0, label="crop_overlap_ratio"
                )
                max_res = gr.Slider(
                    640, 1536, step=64, value=1024, label="downscale_long_side"
                )
                max_segs = gr.Slider(50, 2000, step=50, value=800, label="max_segments")
                max_area = gr.Slider(
                    1000,
                    500000,
                    step=5000,
                    value=150000,
                    label="max_mask_region_area (px)",
                )
                multimask = gr.Checkbox(value=False, label="multimask_output")

            run_btn = gr.Button("Run SAM2 Batch", variant="primary")
            zip_out = gr.File(label="Download segmentation outputs (zip)")

            with gr.Accordion(
                "Shape Filter Parameters (Aspect Ratio Only)", open=False
            ):
                ratio_tol = gr.Slider(
                    minimum=0.05,
                    maximum=1.00,
                    step=0.05,
                    value=0.7,
                    label="ratio_tolerance (± fraction around reference mean)",
                )

            filter_btn = gr.Button(
                "Filter Segments by Shape Reference", variant="secondary"
            )
            zip_filtered = gr.File(label="Download filtered outputs (zip)")

        with gr.Column(scale=2):
            gr.Markdown("### Colorized Segments (Preview)")
            color_gallery = gr.Gallery(columns=3, height=300)
            gr.Markdown("### Label Map (Scaled Grayscale)")
            gray_gallery = gr.Gallery(columns=3, height=300)
            gr.Markdown("### Filtered Previews (After Shape Filtering)")
            filtered_gallery = gr.Gallery(columns=3, height=300)

    # ---------------------------------------------------------------------
    # Event handlers
    # ---------------------------------------------------------------------

    sam_results = {"label_maps": []}

    def _run(
        files_list,
        pps,
        min_px,
        p_iou,
        stab_t,
        crops,
        overlap,
        work_res,
        max_k,
        max_segments,
        max_area_px,
        multi,
        progress=gr.Progress(track_tqdm=True),
    ):
        if not files_list:
            return [], [], None
        try:
            color_pre, gray_pre, label_maps, zip_path = _process_sam_batch(
                files=files_list,
                points_per_side=int(pps),
                min_mask_region_area=int(min_px),
                pred_iou_thresh=float(p_iou),
                stability_score_thresh=float(stab_t),
                crop_n_layers=int(crops),
                crop_overlap_ratio=float(overlap),
                downscale_long_side=int(work_res),
                max_segments=int(max_k),
                multimask_output=bool(multi),
                maximum_mask_region_area=int(max_area_px),
                progress=progress,
            )
            sam_results["label_maps"] = label_maps
            return color_pre, gray_pre, zip_path
        except Exception as e:
            gr.Warning(f"Segmentation error: {type(e).__name__}: {e}")
            return [], [], None

    def _filter(
        _,
        ref_path,
        ratio_tolerance,
        progress=gr.Progress(track_tqdm=True),
    ):
        if not sam_results["label_maps"]:
            gr.Warning("No SAM2 results available. Run segmentation first.")
            return [], None

        try:
            # Use uploaded COCO JSON if provided; otherwise fall back to built-in defaults
            if ref_path:
                loader = CocoEarReferenceLoader(ref_path.name)
                reference_stats = loader.load()
                print(
                    "[REF RATIO] mean=",
                    reference_stats.mean_ratio,
                    "std=",
                    reference_stats.std_ratio,
                    "N=",
                    len(reference_stats.ratios),
                    "(from uploaded file)",
                )
            else:
                print("[REF RATIO] Using built-in defaults (no file uploaded).")
                reference_stats = AspectRatioReferenceStats(
                    mean_ratio=DEFAULT_ASPECT_RATIO_REF["mean_ratio"],
                    std_ratio=DEFAULT_ASPECT_RATIO_REF["std_ratio"],
                    ratios=[DEFAULT_ASPECT_RATIO_REF["mean_ratio"]]
                    * DEFAULT_ASPECT_RATIO_REF["count"],
                )

            filter_service = ShapeFilterService()
            tmpdir = tempfile.mkdtemp(prefix="wheatvision_filtered_")
            zip_path = os.path.join(tmpdir, "filtered_outputs.zip")

            filtered_previews = []
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for idx, label_map in enumerate(sam_results["label_maps"], start=1):
                    progress(
                        (idx, len(sam_results["label_maps"])), desc=f"Filtering {idx}…"
                    )
                    filtered = filter_service.filter_segments(
                        label_map,
                        reference_statistics=reference_stats,
                        ratio_tolerance=float(ratio_tolerance),
                        progress_callback=progress,
                    )

                    vis = _colorize_labels(filtered)
                    filtered_previews.append(vis)

                    base = f"filtered_{idx:03d}"
                    lbl_path = os.path.join(tmpdir, f"{base}_labels_uint16.png")
                    vis_path = os.path.join(tmpdir, f"{base}_labels_color.png")
                    _save_uint16_png(lbl_path, filtered)
                    cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                    zipf.write(lbl_path, os.path.basename(lbl_path))
                    zipf.write(vis_path, os.path.basename(vis_path))

            return filtered_previews, zip_path

        except Exception as e:
            gr.Warning(f"Filtering error: {type(e).__name__}: {e}")
            return [], None

    run_btn.click(
        _run,
        inputs=[
            files,
            points_per_side,
            min_area,
            pred_iou,
            stab,
            crop_layers,
            crop_overlap,
            max_res,
            max_segs,
            max_area,
            multimask,
        ],
        outputs=[color_gallery, gray_gallery, zip_out],
    )

    filter_btn.click(
        _filter,
        inputs=[color_gallery, ref_json, ratio_tol],
        outputs=[filtered_gallery, zip_filtered],
    )
