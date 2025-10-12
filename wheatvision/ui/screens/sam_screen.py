from typing import List, Tuple
import os
import cv2
import gradio as gr
import numpy as np
import tempfile
import zipfile
import torch

from wheatvision.integrations.sam2_adapter import Sam2Adapter, Sam2NotAvailable



def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def _colorize_labels(label_map: np.ndarray) -> np.ndarray:
    h, w = label_map.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    if label_map.max() == 0:
        return vis
    rng = np.random.default_rng(42)
    colors = rng.integers(50, 230, size=(label_map.max() + 1, 3), dtype=np.uint8)
    colors[0] = np.array([255, 255, 255], dtype=np.uint8)  # background
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
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[List[np.ndarray], List[np.ndarray], str]:
    """
    Run SAM2 oversegmentation over a list of filepaths.
    Returns (colorized_previews, label_previews_grayRGB, zip_path).
    """

    adapter = Sam2Adapter()
    if not adapter.is_available():
        reason = getattr(adapter, "availability_error", lambda: None)() or "unknown"
        raise Sam2NotAvailable(f"SAM2 not available: {reason}")

    adapter.build()

    color_previews: List[np.ndarray] = []
    gray_previews: List[np.ndarray] = []

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
                multi_mask_output=bool(multimask_output),
                points_per_batch=64,
                progress_callback=progress,   # fine-grained progress if service uses it
                downscale_long_side_pixels=int(downscale_long_side),
                maximum_segment_count=int(max_segments),
            )

            vis_rgb = _colorize_labels(label_map)            
            disp_rgb = _scaled_gray_for_display(label_map)   

            color_previews.append(vis_rgb)
            gray_previews.append(disp_rgb)

            base = os.path.splitext(os.path.basename(path))[0]
            lbl_name = f"{base}_labels_uint16.png"   
            vis_name = f"{base}_labels_color.png"

            lbl_path = os.path.join(tmpdir, lbl_name)
            vis_path = os.path.join(tmpdir, vis_name)

            _save_uint16_png(lbl_path, label_map)
            cv2.imwrite(vis_path, cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))

            zipf.write(lbl_path, lbl_name)
            zipf.write(vis_path, vis_name)

            # Help VRAM on very large batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return color_previews, gray_previews, zip_path



def build_sam_tab():
    """Batch-only SAM2 oversegmentation UI (Automatic Mask Generator)."""

    gr.Markdown("#### SAM2 Oversegmentation – Batch (Automatic Mask Generator)")

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
                label="Batch images (JPG/PNG)",
                file_types=["image"],
                type="filepath",
            )

            with gr.Accordion("Mask Generator Parameters", open=False):
                points_per_side = gr.Slider(
                    8, 64, step=1, value=32, label="points_per_side (proposal density)",
                    info="Grid prompt density. ↑ = more proposals, better small-part recall; slower & more VRAM. ↓ = faster; may miss tiny parts."
                )
                min_area = gr.Slider(
                    0, 2000, step=50, value=150, label="min_mask_region_area (px)",
                    info="Post-filter by area (px). ↑ = remove tiny specks/noise; may drop small legit parts. ↓ = keep fine details; noisier."
                )
                pred_iou = gr.Slider(
                    0.70, 0.99, step=0.01, value=0.88, label="pred_iou_thresh",
                    info="Quality gate (predicted IoU). ↑ = higher precision/fewer masks. ↓ = higher recall/more low-quality masks."
                )
                stab = gr.Slider(
                    0.80, 0.99, step=0.01, value=0.95, label="stability_score_thresh",
                    info="Robustness gate under threshold perturbations. ↑ = cleaner, fewer artifacts; may drop thin/low-contrast parts."
                )
                crop_layers = gr.Dropdown(
                    choices=[0, 1], value=0, label="crop_n_layers (0 = fastest)",
                    info="Multi-scale crops. 0 = full-frame only (fast). 1 = add cropped pass (better small-object recall; slower)."
                )
                crop_overlap = gr.Slider(
                    0.0, 0.6, step=0.05, value=0.0, label="crop_overlap_ratio",
                    info="Overlap between adjacent crops. Use ≈0.2–0.3 with crop_n_layers=1 to reduce seam misses; 0.0 is fastest."
                )
                max_res = gr.Slider(
                    640, 1536, step=64, value=1024, label="downscale_long_side (working resolution)",
                    info="Resize long side before proposals. ↑ = more detail; slower/VRAM↑. ↓ = faster; may lose thin structures."
                )
                max_segs = gr.Slider(
                    50, 2000, step=50, value=800, label="max_segments cap",
                    info="Max masks to keep after filtering/sorting. Tune to scene complexity to avoid truncation or bloat."
                )
                multimask = gr.Checkbox(
                    value=False, label="multimask_output (more variants per point)",
                    info="Keep multiple candidates per point. On = higher recall/diversity, slower & more overlaps; Off = cleaner/faster."
                )

            run_btn = gr.Button("Run SAM2 Batch", variant="primary")
            zip_out = gr.File(label="Download outputs (zip)")

        with gr.Column(scale=2):
            gr.Markdown("### Colorized segments (preview)")
            color_gallery = gr.Gallery(columns=3, height=300, label="Colorized")

            gr.Markdown("### Label map (scaled grayscale, preview only)")
            gray_gallery = gr.Gallery(columns=3, height=300, label="Label Previews")

    def _run(files_list, pps, min_px, p_iou, stab_t, crops, overlap, work_res, max_k, multi, progress=gr.Progress(track_tqdm=True)):
        if not files_list:
            return [], [], None
        try:
            color_pre, gray_pre, zip_path = _process_sam_batch(
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
                progress=progress,
            )
            return color_pre, gray_pre, zip_path
        except Sam2NotAvailable as e:
            gr.Warning(f"SAM2 not available: {e}")
            return [], [], None
        except Exception as e:
            gr.Warning(f"Segmentation error: {type(e).__name__}: {e}")
            return [], [], None

    run_btn.click(
        _run,
        inputs=[files, points_per_side, min_area, pred_iou, stab, crop_layers, crop_overlap, max_res, max_segs, multimask],
        outputs=[color_gallery, gray_gallery, zip_out],
    )
