import gradio as gr
import os
import tempfile
import zipfile
import cv2
from typing import List


from wheatvision.core.types import PreprocessingConfig, ImageItem
from wheatvision.preprocessing.hsv_masker import HSVForegroundMasker
from wheatvision.preprocessing.row_splitter import RowDensitySplitter
from wheatvision.preprocessing.pipeline import PreprocessingPipeline


def _process_batch(files: List[str], config: PreprocessingConfig):
    """Process a batch of uploaded images and return galleries and a zip."""
    masker = HSVForegroundMasker()
    splitter = RowDensitySplitter()
    pipeline = PreprocessingPipeline(masker=masker, splitter=splitter)
    pipeline.configure(config)

    ears_outputs = []
    stalks_outputs = []
    overlay_outputs = []
    masks_outputs = []

    tmpdir = tempfile.mkdtemp(prefix="wheatvision_")
    zip_path = os.path.join(tmpdir, "preprocessing_outputs.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for path in files or []:
            image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)

            if image_bgr is None:
                continue

            name = os.path.basename(path)
            item = ImageItem(name=name, image_bgr=image_bgr)
            result = pipeline.run_on_item(item)

            # Save ears and stalks
            ears_name = name.rsplit(".", 1)[0] + "_ears.png"
            stalks_name = name.rsplit(".", 1)[0] + "_stalks.png"

            cv2.imwrite(os.path.join(tmpdir, ears_name), result.ears_bgr)
            cv2.imwrite(os.path.join(tmpdir, stalks_name), result.stalks_bgr)

            zipf.write(os.path.join(tmpdir, ears_name), ears_name)
            zipf.write(os.path.join(tmpdir, stalks_name), stalks_name)

            overlay = image_bgr.copy()
            cv2.line(
                overlay,
                (0, result.cut_position_y),
                (overlay.shape[1], result.cut_position_y),
                (0, 0, 255),
                2,
            )

            mask_rgb = cv2.cvtColor(result.foreground_mask, cv2.COLOR_GRAY2RGB)

            overlay_outputs.append(overlay[:, :, ::-1])  # BGR->RGB for Gradio
            ears_outputs.append(result.ears_bgr[:, :, ::-1])
            stalks_outputs.append(result.stalks_bgr[:, :, ::-1])
            masks_outputs.append(mask_rgb)

    return overlay_outputs, masks_outputs, ears_outputs, stalks_outputs, zip_path


def build_preprocessing_tab():
    """Build the preprocessing UI tab."""
    with gr.Row():
        with gr.Column(scale=1):
            files = gr.Files(
                label="Batch images (JPG/PNG)", file_types=["image"], type="filepath"
            )

            with gr.Accordion("Masking Parameters", open=False):
                hue_min = gr.Slider(0, 180, value=0, step=1, label="Hue min")
                hue_max = gr.Slider(0, 180, value=180, step=1, label="Hue max")
                saturation_max_background = gr.Slider(
                    0, 255, value=40, step=1, label="Saturation max (background)"
                )
                value_min_background = gr.Slider(
                    0, 255, value=150, step=1, label="Value min (background)"
                )

                open_kernel = gr.Slider(1, 15, value=6, step=1, label="Open kernel")
                open_iterations = gr.Slider(
                    0, 5, value=1, step=1, label="Open iterations"
                )
                close_kernel = gr.Slider(1, 15, value=5, step=1, label="Close kernel")
                close_iterations = gr.Slider(
                    0, 5, value=1, step=1, label="Close iterations"
                )

            with gr.Accordion("Split Parameters", open=False):
                top_fraction = gr.Slider(
                    0.0, 0.5, value=0.15, step=0.01, label="Top search start fraction"
                )
                bottom_fraction = gr.Slider(
                    0.5, 1.0, value=0.85, step=0.01, label="Bottom search end fraction"
                )
                gaussian_sigma = gr.Slider(
                    0.0, 20.0, value=8.0, step=0.5, label="Gaussian sigma"
                )
                min_fraction = gr.Slider(
                    0.0, 1.0, value=0.25, step=0.01, label="Min cut fraction clamp"
                )
                max_fraction = gr.Slider(
                    0.0, 1.0, value=0.75, step=0.01, label="Max cut fraction clamp"
                )
                margin_pixels = gr.Slider(
                    0, 50, value=10, step=1, label="Margin pixels below valley"
                )

            run_btn = gr.Button("Run Preprocessing", variant="primary")
            zip_out = gr.File(label="Download outputs (zip)")

        with gr.Column(scale=2):

            gr.Markdown("### Overlay with cut line")
            overlay_gallery = gr.Gallery(columns=3, height=240, label="Overlays")

            gr.Markdown("### Foreground Masks")
            masks_gallery = gr.Gallery(columns=3, height=240, label="Masks")

            gr.Markdown("### Ears and Stalks")
            ears_gallery = gr.Gallery(columns=3, height=240, label="Ears")
            stalks_gallery = gr.Gallery(columns=3, height=240, label="Stalks")

    def _collect_config(*vals) -> PreprocessingConfig:
        """Collect UI slider values into a configuration object."""
        (
            hue_min_v,
            hue_max_v,
            sat_max_v,
            val_min_v,
            open_kernel_v,
            open_iterations_v,
            close_kernel_v,
            close_iterations_v,
            top_fraction_v,
            bottom_fraction_v,
            gaussian_sigma_v,
            min_fraction_v,
            max_fraction_v,
            margin_pixels_v,
        ) = vals

        return PreprocessingConfig(
            hsv=__import__(
                "wheatvision.core.types", fromlist=["HSVThresholds"]
            ).HSVThresholds(
                hue_min=int(hue_min_v),
                hue_max=int(hue_max_v),
                saturation_max_background=int(sat_max_v),
                value_min_background=int(val_min_v),
            ),
            morphology=__import__(
                "wheatvision.core.types", fromlist=["MorphologyConfig"]
            ).MorphologyConfig(
                open_kernel=int(open_kernel_v),
                open_iterations=int(open_iterations_v),
                close_kernel=int(close_kernel_v),
                close_iterations=int(close_iterations_v),
            ),
            split=__import__(
                "wheatvision.core.types", fromlist=["SplitSearchConfig"]
            ).SplitSearchConfig(
                top_fraction=float(top_fraction_v),
                bottom_fraction=float(bottom_fraction_v),
                gaussian_sigma=float(gaussian_sigma_v),
                min_fraction=float(min_fraction_v),
                max_fraction=float(max_fraction_v),
                margin_pixels=int(margin_pixels_v),
            ),
        )

    def _run(files_list, *params):
        """Glue function to run preprocessing with UI params."""
        cfg = _collect_config(*params)
        overlays, masks, ears, stalks, zip_path = _process_batch(files_list, cfg)
        return overlays, masks, ears, stalks, zip_path

    run_btn.click(
        _run,
        inputs=[
            files,
            hue_min,
            hue_max,
            saturation_max_background,
            value_min_background,
            open_kernel,
            open_iterations,
            close_kernel,
            close_iterations,
            top_fraction,
            bottom_fraction,
            gaussian_sigma,
            min_fraction,
            max_fraction,
            margin_pixels,
        ],
        outputs=[overlay_gallery, masks_gallery, ears_gallery, stalks_gallery, zip_out],
    )
