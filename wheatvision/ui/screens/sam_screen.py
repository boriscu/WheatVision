# wheatvision/ui/screens/sam_screen.py
from typing import Tuple
import cv2
import gradio as gr
import numpy as np

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


def build_sam_tab():
    gr.Markdown("#### SAM2 Oversegmentation (Automatic Mask Generator)")

    adapter = Sam2Adapter()
    if not adapter.is_available():
        reason = getattr(adapter, "availability_error", lambda: None)() or "unknown"
        gr.Markdown(
            f"> ⚠️ **SAM2 isn’t installed or not importable.** ({reason})  \n"
            "Install SAM2 and configure your `.env`."
        )
        return

    with gr.Row():
        image_in = gr.Image(
            label="Upload image",
            type="numpy",
            sources=["upload", "clipboard"],
            height=480,
        )
        with gr.Column():
            run_btn = gr.Button("Run Segmentation", variant="primary")
            status = gr.Markdown("Ready.")

            with gr.Accordion("Advanced (AMG)", open=False):
                points_per_side = gr.Slider(
                    minimum=8, maximum=64, step=1, value=32,
                    label="points_per_side (proposal density)"
                )
                min_area = gr.Slider(
                    minimum=0, maximum=2000, step=50, value=150,
                    label="min_mask_region_area (px)"
                )
                pred_iou = gr.Slider(
                    minimum=0.70, maximum=0.99, step=0.01, value=0.88,
                    label="pred_iou_thresh"
                )
                stab = gr.Slider(
                    minimum=0.80, maximum=0.99, step=0.01, value=0.95,
                    label="stability_score_thresh"
                )
                crop_layers = gr.Dropdown(
                    choices=[0, 1], value=0, label="crop_n_layers (0 = fastest)"
                )
                crop_overlap = gr.Slider(
                    minimum=0.0, maximum=0.6, step=0.05, value=0.0,
                    label="crop_overlap_ratio"
                )
                max_res = gr.Slider(
                    minimum=640, maximum=1536, step=64, value=1024,
                    label="downscale_long_side (working resolution)"
                )
                max_segs = gr.Slider(
                    minimum=50, maximum=2000, step=50, value=800,
                    label="max_segments cap"
                )
                multimask = gr.Checkbox(
                    value=False, label="multimask_output (more variants per point)"
                )

    with gr.Row():
        vis_out = gr.Image(label="Colorized segments (preview)", type="numpy", height=480)
        label_out = gr.Image(label="Label map (uint16 PNG, 0=background)", type="numpy", height=480)

    def run_overseg(
        img: np.ndarray,
        pps: int,
        min_px: int,
        p_iou: float,
        stab_t: float,
        crops: int,
        overlap: float,
        work_res: int,
        max_k: int,
        multi: bool,
        progress=gr.Progress(track_tqdm=True),
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        if img is None:
            return None, None, "Please upload an image."
        try:
            label_map = adapter.oversegment(
                image_bgr=_to_uint8(img),
                # AMG knobs
                points_per_side=int(pps),
                pred_iou_thresh=float(p_iou),
                stability_score_thresh=float(stab_t),
                box_nms_thresh=0.7,
                crop_n_layers=int(crops),
                crop_overlap_ratio=float(overlap),
                min_mask_region_area=int(min_px),
                multimask_output=bool(multi),
                points_per_batch=64,
                # app-level controls
                progress=progress,
                downscale_long_side=int(work_res),
                max_segments=int(max_k),
            )
        except Sam2NotAvailable as e:
            return None, None, f"SAM2 not available: {e}"
        except Exception as e:
            return None, None, f"Segmentation error: {type(e).__name__}: {e}"

        vis = _colorize_labels(label_map)

        max_id = int(label_map.max())
        if max_id <= 0:
            disp = np.zeros_like(label_map, dtype=np.uint8)
        else:
            scale = max(1, 255 // max_id)
            disp = (label_map * scale).clip(0, 255).astype(np.uint8)

        return vis, disp, f"Done. Segments: {max_id}"

    run_btn.click(
        run_overseg,
        inputs=[image_in, points_per_side, min_area, pred_iou, stab, crop_layers, crop_overlap, max_res, max_segs, multimask],
        outputs=[vis_out, label_out, status],
    )
