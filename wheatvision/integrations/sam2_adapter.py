import os
from contextlib import contextmanager
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from typing import List, Optional, Tuple
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
        grid_step: int = 64,
        iou_threshold: float = 0.8,
        multimask: bool = True,
        progress=None,                
        downscale_long_side: int | None = 1024,  
        max_segments: int = 600,       
        min_area: int = 100,           
        time_budget_s: float | None = None,  
    ) -> np.ndarray:
        """Oversegment by probing a uniform grid of positive points. Returns (H,W) label map (int32)."""
        import time, cv2

        if self._predictor is None:
            self.build()
        assert self._predictor is not None

        t0 = time.perf_counter()
        orig_h, orig_w = image_bgr.shape[:2]
        scale = 1.0
        work = image_bgr
        if downscale_long_side and max(orig_h, orig_w) > downscale_long_side:
            scale = downscale_long_side / float(max(orig_h, orig_w))
            new_w = int(round(orig_w * scale))
            new_h = int(round(orig_h * scale))
            work = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        image_rgb = work[..., ::-1].copy()
        self._predictor.set_image(image_rgb)

        H, W = image_rgb.shape[:2]
        ys = np.arange(grid_step // 2, H, grid_step)
        xs = np.arange(grid_step // 2, W, grid_step)
        total_pts = max(1, len(xs) * len(ys))

        kept_masks: list[np.ndarray] = []

        def iou_with_existing(mask: np.ndarray) -> float:
            best = 0.0
            for m in kept_masks:
                inter = np.logical_and(mask, m).sum()
                if inter == 0:
                    continue
                union = mask.sum() + m.sum() - inter
                if union == 0:
                    continue
                r = inter / union
                if r > best:
                    best = r
                    if best >= iou_threshold:
                        break
            return best

        use_autocast = bool(self._cfg.autocast if self._cfg else True)
        use_cuda = self._predictor.model.device.type == "cuda"
        ctx = self._autocast_ctx(enabled=(use_autocast and use_cuda))

        print(f"[SAM2] device={self._predictor.model.device} size={W}x{H} step={grid_step} multimask={multimask} pts={total_pts}", flush=True)

        with ctx:
            idx = 0
            for y in ys:
                for x in xs:
                    idx += 1
                    if progress:
                        progress(min(0.99, idx / total_pts), f"Probing point {idx}/{total_pts}, kept={len(kept_masks)}")

                    pts = np.array([[int(x), int(y)]], dtype=np.int32)
                    labels = np.array([1], dtype=np.int32)
                    masks, scores, _ = self._predictor.predict(
                        point_coords=pts,
                        point_labels=labels,
                        multimask_output=multimask,
                    )
                    if scores is not None:
                        order = np.argsort(-scores)
                        masks = masks[order]

                    for m in masks:
                        m_bool = m.astype(bool)
                        if min_area and m_bool.sum() < min_area:
                            continue
                        if iou_with_existing(m_bool) >= iou_threshold:
                            continue
                        kept_masks.append(m_bool)
                        if len(kept_masks) >= max_segments:
                            break

                    if len(kept_masks) >= max_segments:
                        break
                    if time_budget_s is not None and (time.perf_counter() - t0) > time_budget_s:
                        print("[SAM2] time budget reached; stopping early", flush=True)
                        break
                else:
                    continue
                break

        label_map = np.zeros((H, W), dtype=np.int32)
        for i, m in enumerate(kept_masks, start=1):
            label_map[m] = i

        if scale != 1.0:
            label_map = cv2.resize(label_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        if progress:
            progress(1.0, f"Done. segments={int(label_map.max())}")
        print(f"[SAM2] done: segments={int(label_map.max())}", flush=True)
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
