import os
from contextlib import contextmanager
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

        model = build_sam2(str(cfg["model_cfg"]), str(cfg["checkpoint"]))
        predictor = SAM2ImagePredictor(model)

        device = cfg["device"]
        predictor.model.to(device)
        self._predictor = predictor

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
        repo = Path(
            cfg.sam2_repo_root or os.getenv("WHEATVISION_SAM2_REPO", "sam2")
        ).resolve()
        ckpt = Path(
            cfg.checkpoint_path or os.getenv("WHEATVISION_SAM2_CKPT", "")
        ).resolve()
        model_cfg = Path(
            cfg.model_cfg_path or os.getenv("WHEATVISION_SAM2_CFG", "")
        ).resolve()
        device = (
            cfg.device
            or os.getenv("WHEATVISION_SAM2_DEVICE")
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if not ckpt.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt}\n"
                "Set Sam2Config.checkpoint_path or WHEATVISION_SAM2_CKPT."
            )
        if not model_cfg.is_file():
            rel_cfg = repo / model_cfg
            if rel_cfg.is_file():
                model_cfg = rel_cfg.resolve()
            else:
                raise FileNotFoundError(
                    f"Model config not found: {model_cfg}\n"
                    "Set Sam2Config.model_cfg_path or WHEATVISION_SAM2_CFG (can be relative to repo)."
                )

        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        return {
            "repo": repo,
            "checkpoint": ckpt,
            "model_cfg": model_cfg,
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
