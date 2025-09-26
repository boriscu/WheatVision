# wheatvision/ui/screens/sam_screen.py
from typing import List, Tuple

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


def build_sam_tab():
    gr.Markdown("#### SAM2 segmentation")

    adapter = Sam2Adapter()

    # If SAM2 is not importable/configured, show a minimal fallback UI (Auto Mask).
    if not adapter.is_available():
        gr.Markdown(
            "> ⚠️ **SAM2 isn’t installed or not importable.** "
            "Follow the README to install SAM2 and configure `.env`. "
            "Until then, you can use the simple Auto Mask baseline below."
        )

        with gr.Row():
            image_in = gr.Image(
                label="Upload image",
                type="numpy",
                sources=["upload", "clipboard"],
                height=420,
            )
            with gr.Column():
                status = gr.Markdown("SAM2 unavailable — using baseline.")
                auto_btn = gr.Button("Auto Mask (Otsu)")

        mask_out = gr.Image(label="Mask", type="numpy", height=420)

        def auto_mask_baseline(img: np.ndarray):
            if img is None:
                return None, "Please upload an image."
            gray = cv2.cvtColor(_to_uint8(img), cv2.COLOR_BGR2GRAY)
            thr, binm = cv2.threshold(
                gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV
            )
            vis_rgb = np.stack([binm, binm, binm], axis=-1)
            return vis_rgb, f"Auto mask (Otsu) thr={thr:.1f}"

        auto_btn.click(
            auto_mask_baseline, inputs=[image_in], outputs=[mask_out, status]
        )
        return

    # SAM2 available: render full point-prompt UI + optional Auto Mask baseline.
    with gr.Row():
        image_in = gr.Image(
            label="Upload image",
            type="numpy",
            sources=["upload", "clipboard"],
            height=420,
        )
        with gr.Column():
            status = gr.Markdown("Ready.")
            fg_points_state = gr.State(value=[])  # type: List[Tuple[int,int]]
            bg_points_state = gr.State(value=[])

            add_mode = gr.Radio(
                ["Foreground (+)", "Background (−)"],
                value="Foreground (+)",
                label="Click mode",
            )
            with gr.Row():
                clear_btn = gr.Button("Clear points")
                run_btn = gr.Button("Run Point Segmentation", variant="primary")

            auto_btn = gr.Button("Auto Mask (Otsu)")

    mask_out = gr.Image(label="Mask", type="numpy", height=420)

    # Collect clicks to build FG/BG prompt sets
    # NOTE: evt comes LAST and is injected by Gradio automatically; DO NOT include gr.EventData() in inputs.
    def on_click(
        img: np.ndarray,
        mode: str,
        fg_pts: List[Tuple[int, int]],
        bg_pts: List[Tuple[int, int]],
        evt: gr.SelectData,  # injected automatically
    ):
        if img is None:
            return fg_pts, bg_pts, "Upload an image first."
        # evt.index returns (x, y)
        x, y = int(evt.index[0]), int(evt.index[1])
        if mode.startswith("Foreground"):
            fg_pts = fg_pts + [(x, y)]
        else:
            bg_pts = bg_pts + [(x, y)]
        return fg_pts, bg_pts, f"Points → FG:{len(fg_pts)} / BG:{len(bg_pts)}"

    image_in.select(
        fn=on_click,
        inputs=[image_in, add_mode, fg_points_state, bg_points_state],
        outputs=[fg_points_state, bg_points_state, status],
    )

    def on_clear():
        return [], [], "Cleared points."

    clear_btn.click(
        on_clear, inputs=None, outputs=[fg_points_state, bg_points_state, status]
    )

    # Run SAM2 with collected points
    def run_points(
        img: np.ndarray, fg_pts: List[Tuple[int, int]], bg_pts: List[Tuple[int, int]]
    ):
        if img is None:
            return None, "Please upload an image."
        try:
            mask01 = adapter.predict_from_points(img, fg_pts, bg_pts)  # (H,W) {0,1}
        except Sam2NotAvailable as e:
            return None, f"SAM2 not available: {e}"
        vis = (mask01 * 255).astype(np.uint8)
        vis_rgb = np.stack([vis, vis, vis], axis=-1)
        return vis_rgb, f"Done. FG:{len(fg_pts)} / BG:{len(bg_pts)}"

    run_btn.click(
        run_points,
        inputs=[image_in, fg_points_state, bg_points_state],
        outputs=[mask_out, status],
    )

    # Optional quick baseline even when SAM2 is available
    def auto_mask(img: np.ndarray):
        if img is None:
            return None, "Please upload an image."
        gray = cv2.cvtColor(_to_uint8(img), cv2.COLOR_BGR2GRAY)
        thr, binm = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        vis_rgb = np.stack([binm, binm, binm], axis=-1)
        return vis_rgb, f"Auto mask (Otsu) thr={thr:.1f}"

    auto_btn.click(auto_mask, inputs=[image_in], outputs=[mask_out, status])
