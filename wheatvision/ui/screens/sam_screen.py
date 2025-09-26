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
    """
    Turn a (H,W) int label map (0..N) into an RGB visualization.
    Background=0 -> white background for this app (can swap to black if you prefer).
    """
    h, w = label_map.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    if label_map.max() == 0:
        return vis  # empty

    # Generate deterministic colors
    rng = np.random.default_rng(42)
    colors = rng.integers(
        low=50, high=230, size=(label_map.max() + 1, 3), dtype=np.uint8
    )
    colors[0] = np.array([255, 255, 255], dtype=np.uint8)  # background white

    vis = colors[label_map]
    return vis


def build_sam_tab():
    gr.Markdown("#### SAM2 Oversegmentation")

    adapter = Sam2Adapter()

    if not adapter.is_available():
        gr.Markdown(
            "> ⚠️ **SAM2 isn’t installed or not importable.** "
            "Install SAM2 and configure your `.env` as documented. "
            "This tab requires SAM2."
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
            grid_step = gr.Slider(
                minimum=24,
                maximum=160,
                value=64,
                step=8,
                label="Grid step (px) — smaller → more segments, slower",
            )
            iou_thresh = gr.Slider(
                minimum=0.5,
                maximum=0.95,
                value=0.8,
                step=0.05,
                label="Deduplicate IoU threshold",
            )
            multimask = gr.Checkbox(
                value=True,
                label="Use multimask per point (more variants per point)",
            )
            run_btn = gr.Button("Run Segmentation", variant="primary")
            status = gr.Markdown("Ready.")

    with gr.Row():
        vis_out = gr.Image(
            label="Colorized segments (preview)", type="numpy", height=480
        )
        label_out = gr.Image(
            label="Label map (uint16 PNG, 0=background)",
            type="numpy",
            height=480,
        )

    def run_overseg(
        img: np.ndarray, step: int, iou_t: float, multi: bool
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        if img is None:
            return None, None, "Please upload an image."

        try:
            # Returns a (H,W) integer label map
            label_map = adapter.oversegment(
                image_bgr=_to_uint8(img),
                grid_step=int(step),
                iou_threshold=float(iou_t),
                multimask=bool(multi),
            )
        except Sam2NotAvailable as e:
            return None, None, f"SAM2 not available: {e}"
        except Exception as e:
            return None, None, f"Segmentation error: {e}"

        vis = _colorize_labels(label_map)

        # Ensure label map is a compact dtype for saving/preview (uint16 to be safe for many segments)
        if label_map.dtype != np.uint16:
            if label_map.max() < 65535:
                label_map = label_map.astype(np.uint16)
            else:
                # fallback if unexpected huge label count
                label_map = (label_map % 65535).astype(np.uint16)

        return vis, label_map, f"Done. Segments: {int(label_map.max())}"

    run_btn.click(
        run_overseg,
        inputs=[image_in, grid_step, iou_thresh, multimask],
        outputs=[vis_out, label_out, status],
    )
