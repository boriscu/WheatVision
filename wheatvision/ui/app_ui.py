import base64
from pathlib import Path
import gradio as gr

from wheatvision.ui.screens.preprocessing_screen import build_preprocessing_tab
from wheatvision.ui.screens.sam_screen import build_sam_tab


def _data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_app() -> gr.Blocks:
    logo_path = Path(__file__).parent / "assets" / "logo.png"
    logo_src = _data_uri(logo_path)

    with gr.Blocks(title="WheatVision") as demo:
        gr.HTML(
            f"""
                <div style="text-align:center; margin: 6px 0 10px;">
                <img src="{logo_src}" alt="Company Logo"
                    style="display:block; margin:0 auto; height:60px; width:auto;" />
                </div>
            """
        )
        gr.Markdown("# 🌾 WheatVision — Ears/Stalks Toolkit")

        with gr.Tabs():
            with gr.TabItem("Preprocessing"):
                build_preprocessing_tab()
            with gr.TabItem("Segmentation"):
                build_sam_tab()

        gr.Markdown(
            "Made for upright, white-background wheat images • Future: SAM2-based refinement"
        )
    return demo
