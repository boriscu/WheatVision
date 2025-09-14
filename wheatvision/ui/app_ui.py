import gradio as gr

from wheatvision.ui.screens.preprocessing_screen import build_preprocessing_tab
from wheatvision.ui.screens.sam_screen import build_sam_tab


def build_app() -> gr.Blocks:
    """Build the multi-screen Gradio application."""
    with gr.Blocks(title="WheatVision") as demo:
        gr.Markdown("# 🌾 WheatVision — Ears/Stalks Toolkit")
        with gr.Tabs():
            with gr.TabItem("Preprocessing"):
                build_preprocessing_tab()
            with gr.TabItem("Segmentation (Coming Soon)"):
                build_sam_tab()
        gr.Markdown(
            "Made for upright, white-background wheat images • Future: SAM2-based refinement",
        )
    return demo
