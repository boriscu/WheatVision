from contextlib import contextmanager
from dotenv import load_dotenv, find_dotenv
from typing import List, Optional, Tuple
import importlib
import numpy as np
import torch

from wheatvision.core.types import Sam2Config


class Sam2NotAvailable(RuntimeError):
    """Raised when SAM2 is not importable/installed."""


class Sam2Adapter:
    """
    Thin OOP wrapper around SAM2's image predictor:
      - lazy import & construction
      - explicit configure() step
      - point-based prediction as a method
      - no free functions
    """

    def __init__(self) -> None:
        load_dotenv(find_dotenv(filename=".env", usecwd=True))  
        self._cfg: Optional[Sam2Config] = None
        self._predictor = None
        self._available = self._try_import_sam2()

    def is_available(self) -> bool:
        """True if SAM2 is importable in this environment."""
        return self._available

    def configure(self, config: Sam2Config) -> None:
        """
        Store configuration and reset any existing predictor.
        Call build() afterward (or let it build lazily on first use).
        """
        self._cfg = config
        self._predictor = None

    def build(self) -> None:
        """
        Build the underlying SAM2 predictor immediately.
        If not called, it will be built lazily upon first prediction.
        """
        if not self._available:
            raise Sam2NotAvailable(
                "SAM2 is not importable. Ensure `pip install -e ./sam2` succeeded."
            )
        if self._cfg is None:
            self._cfg = Sam2Config()

        cfg = self._resolve_config(self._cfg)
        build_sam2, SAM2ImagePredictor = self._get_sam2_constructors()

        model = build_sam2(str(cfg["model_cfg_name"]), str(cfg["checkpoint"]))
        predictor = SAM2ImagePredictor(model)

        device = cfg["device"]
        predictor.model.to(device)
        self._predictor = predictor
    
    @torch.inference_mode()
    def oversegment(
        self,
        image_bgr: np.ndarray,
        points_per_side: int = 48,
        pred_iou_thresh: float = 0.75,
        stability_score_thresh: float = 0.90,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 1,
        crop_overlap_ratio: float = 0.2,
        min_mask_region_area: int = 80,
        multimask_output: bool = True,
        points_per_batch: int = 64,
        progress=None,
        downscale_long_side: int | None = 1024,
        max_segments: int = 1500,
    ) -> np.ndarray:
        import cv2, numpy as np

        if self._predictor is None:
            self.build()
        assert self._predictor is not None

        # optional downscale
        H0, W0 = image_bgr.shape[:2]
        scale = 1.0
        work = image_bgr
        if downscale_long_side and max(H0, W0) > downscale_long_side:
            scale = downscale_long_side / float(max(H0, W0))
            work = cv2.resize(
                image_bgr,
                (int(round(W0 * scale)), int(round(H0 * scale))),
                interpolation=cv2.INTER_AREA,
            )
        img_rgb = work[..., ::-1].copy()

        AMG = importlib.import_module("sam2.automatic_mask_generator").SAM2AutomaticMaskGenerator

        def run_amg(pp_side, iou_t, stab_t, crops, overlap, min_area, multi):
            amg = AMG(
                self._predictor.model,
                points_per_side=pp_side,
                points_per_batch=points_per_batch,
                pred_iou_thresh=iou_t,
                stability_score_thresh=stab_t,
                mask_threshold=0.0,
                box_nms_thresh=box_nms_thresh,
                crop_n_layers=crops,
                crop_overlap_ratio=overlap,
                min_mask_region_area=min_area,
                output_mode="binary_mask",
                multimask_output=multi,
            )
            if progress: progress(0.05, "Generating proposals (AMG)…")
            return amg.generate(img_rgb)

        # pass 1 (default)
        props = run_amg(points_per_side, pred_iou_thresh, stability_score_thresh,
                        crop_n_layers, crop_overlap_ratio, min_mask_region_area, multimask_output)

        # if too few, relax and densify once
        if len(props) <= 3:
            if progress: progress(0.10, "Few masks; retrying with denser/looser settings…")
            props = run_amg(
                pp_side=max(64, points_per_side),
                iou_t=0.70,
                stab_t=0.88,
                crops=max(1, crop_n_layers),
                overlap=max(0.2, crop_overlap_ratio),
                min_area=max(20, min_mask_region_area // 2),
                multi=True,
            )

        if not props:
            if progress: progress(1.0, "No segments found.")
            return np.zeros((H0, W0), dtype=np.int32)

        # sort by area/score and cap
        props.sort(key=lambda d: (d.get("area", 0), d.get("predicted_iou", 0.0)), reverse=True)
        if max_segments and len(props) > max_segments:
            props = props[:max_segments]

        H, W = img_rgb.shape[:2]
        label_map = np.zeros((H, W), dtype=np.int32)
        kept = 0
        for i, d in enumerate(props, 1):
            m = d["segmentation"]
            m_bool = (m > 0) if isinstance(m, np.ndarray) else np.array(m, dtype=bool)
            if m_bool.sum() < min_mask_region_area:
                continue
            kept += 1
            label_map[m_bool] = kept
            if progress and (i % 25 == 0):
                progress(min(0.98, i / max(1, len(props))), f"Painting {i}/{len(props)}")

        # upsample back
        if scale != 1.0:
            label_map = cv2.resize(label_map, (W0, H0), interpolation=cv2.INTER_NEAREST)

        if progress: progress(1.0, f"Done (AMG). segments={int(label_map.max())}")
        print(f"[SAM2] AMG segments={int(label_map.max())}", flush=True)
        return label_map
            
    @torch.inference_mode()
    def predict_from_points(
        self,
        image_bgr: np.ndarray,
        fg_points: List[Tuple[int, int]] | None = None,
        bg_points: List[Tuple[int, int]] | None = None,
    ) -> np.ndarray:
        """
        Run point-prompt segmentation. Returns a (H,W) uint8 mask in {0,1}.
        """
        if self._predictor is None:
            self.build()

        assert self._predictor is not None
        image_rgb = image_bgr[..., ::-1].copy()  # BGR -> RGB
        self._predictor.set_image(image_rgb)

        points, labels = self._pack_points(fg_points, bg_points)

        use_autocast = bool(self._cfg.autocast if self._cfg else True)
        use_cuda = self._predictor.model.device.type == "cuda"
        ctx = self._autocast_ctx(enabled=(use_autocast and use_cuda))

        with ctx:
            masks, scores, _ = self._predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=False,
            )

        mask = masks[0].astype(np.uint8)  # (H, W) {0,1}
        return mask

    def _try_import_sam2(self) -> bool:
        try:
            import sam2  # noqa: F401

            return True
        except Exception:
            return False

    def _get_sam2_constructors(self):
        if not self._available:
            raise Sam2NotAvailable("SAM2 import failed.")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        return build_sam2, SAM2ImagePredictor

    def _resolve_config(self, cfg: Sam2Config) -> dict:
        from pathlib import Path
        import os, torch

        repo = Path(cfg.sam2_repo_root or os.getenv("WHEATVISION_SAM2_REPO", "external/sam2_repo")).resolve()

        ckpt_str = cfg.checkpoint_path or os.getenv("WHEATVISION_SAM2_CKPT", "")
        if not ckpt_str:
            raise FileNotFoundError("WHEATVISION_SAM2_CKPT is empty / missing in .env.")
        ckpt = Path(ckpt_str)
        if not ckpt.is_absolute():
            ckpt = (Path.cwd() / ckpt).resolve()
        if not ckpt.is_file():
            alt = (repo / ckpt_str).resolve()
            if alt.is_file():
                ckpt = alt
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found at:\n  {ckpt}\n"
                    f"(also tried repo-relative: {alt})\n"
                    "Fix WHEATVISION_SAM2_CKPT in .env."
                )

        cfg_str = (cfg.model_cfg_path or os.getenv("WHEATVISION_SAM2_CFG", "")).strip()
        if not cfg_str:
            raise FileNotFoundError("WHEATVISION_SAM2_CFG is empty / missing in .env.")

        cfg_path = Path(cfg_str)
        if not cfg_path.is_absolute():
            cfg_path = (Path.cwd() / cfg_path).resolve()
        if not cfg_path.is_file():
            alt = (repo / (cfg_str.lstrip("./"))).resolve()
            if alt.is_file():
                cfg_path = alt

        if cfg_path.is_file():
 
            try:
                p = cfg_path.as_posix()
                needle = "/sam2/configs/"
                i = p.rfind(needle)
                if i != -1:
                    model_cfg_name = "configs/" + p[i + len(needle):]
                else:
                    needle2 = "/configs/"
                    j = p.rfind(needle2)
                    if j != -1:
                        model_cfg_name = "configs/" + p[j + len(needle2):]
                    else:
                        model_cfg_name = cfg_path.name
            except Exception:
                model_cfg_name = cfg_path.name
        else:
            model_cfg_name = cfg_str

        device = (
            cfg.device
            or os.getenv("WHEATVISION_SAM2_DEVICE")
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        return {
            "repo": repo,
            "checkpoint": ckpt,
            "model_cfg_name": model_cfg_name,
            "device": device,
        }

    @staticmethod
    def _pack_points(
        fg_points: List[Tuple[int, int]] | None,
        bg_points: List[Tuple[int, int]] | None,
    ):
        import numpy as np

        fg_points = fg_points or []
        bg_points = bg_points or []
        if not fg_points and not bg_points:
            return None, None

        pts = np.array(fg_points + bg_points, dtype=np.int32)
        labels = np.array([1] * len(fg_points) + [0] * len(bg_points), dtype=np.int32)
        return pts, labels

    @contextmanager
    def _autocast_ctx(self, enabled: bool):
        if enabled:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                yield
        else:
            yield
