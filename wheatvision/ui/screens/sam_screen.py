import gradio as gr


def build_sam_tab():
    """Render a placeholder for future SAM2 segmentation."""
    gr.Markdown(
        "#### SAM2 segmentation will live here with prompts, masks, and refinement.",
    )
    gr.Markdown("> ⚠️ This screen is a placeholder for the upcoming SAM2 step.")
