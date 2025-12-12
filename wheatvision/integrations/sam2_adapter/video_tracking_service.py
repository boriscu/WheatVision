import os
import torch
import numpy as np
import cv2
from typing import List, Optional, Dict, Any, Tuple
from tempfile import TemporaryDirectory

# SAM2 Imports - dependent on the external structure
try:
    from external.sam2_repo.sam2.build_sam import build_sam2_video_predictor
except ImportError:
    from sam2.build_sam import build_sam2_video_predictor

class VideoTrackingService:
    """
    Manages SAM2 video tracking state and operations.
    """
    def __init__(self, model_config_name: str, checkpoint_path: str, device: str = "cuda"):
        """
        Initializes the VideoTrackingService with configuration paths.
        
        Args:
            model_config_name (str): Hydra config name (e.g. "configs/sam2.1/sam2.1_hiera_s.yaml").
            checkpoint_path (str): Absolute path to the SAM2 checkpoint file.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.model_config_name = model_config_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.predictor = None
        self.inference_state = None
        self.video_dir_ctx = None
        self.video_dir = None
        
    def _ensure_predictor(self):
        """
        Ensures the SAM2 video predictor is built.
        
        Raises:
            FileNotFoundError: If the checkpoint path does not exist.
        """
        if self.predictor is None:
            if not os.path.exists(self.checkpoint_path):
                 raise FileNotFoundError(f"SAM2 checkpoint not found at {self.checkpoint_path}")
            
            # build_sam2_video_predictor handles the config loading via hydra internally
            self.predictor = build_sam2_video_predictor(
                self.model_config_name, 
                self.checkpoint_path, 
                device=self.device
            )

    def init_video(self, frames: List[np.ndarray]):
        """
        Initializes the video predictor with a sequence of frames.
        
        This writes frames to a temporary directory as SAM2 requires JPEG files.
        SAM2 reads by sorted filename.
        
        Args:
            frames (List[np.ndarray]): List of video frames as numpy arrays (BGR).
        """
        self._ensure_predictor()
        
        # Cleanup previous context if exists
        if self.video_dir_ctx:
            self.video_dir_ctx.cleanup()
            
        self.video_dir_ctx = TemporaryDirectory(prefix="wheatvision_sam2_vid_")
        self.video_dir = self.video_dir_ctx.name
        
        for idx, frame in enumerate(frames):
            path = os.path.join(self.video_dir, f"{idx:05d}.jpg")
            cv2.imwrite(path, frame)
            
        self.inference_state = self.predictor.init_state(video_path=self.video_dir)
        self.frames_count = len(frames)
        self.frames_cache = frames 

    def add_mask(self, frame_idx: int, mask: np.ndarray, obj_id: int):
        """
        Adds an initial mask for a specific object ID at a given frame.
        
        Args:
            frame_idx (int): The index of the frame.
            mask (np.ndarray): The binary mask to add.
            obj_id (int): The unique identifier for the object.
            
        Raises:
            RuntimeError: If video has not been initialized.
        """
        if self.inference_state is None:
            raise RuntimeError("Video not initialized. Call init_video first.")
            
        _, _, _ = self.predictor.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            mask=mask
        )

    def add_click(self, frame_idx: int, points: np.ndarray, labels: np.ndarray, obj_id: int):
        """
        Adds a click interaction (positive/negative) for a specific object on a frame.
        
        Args:
            frame_idx (int): The index of the frame.
            points (np.ndarray): (N, 2) array of coordinates.
            labels (np.ndarray): (N,) array of 1 (positive) or 0 (negative).
            obj_id (int): The unique identifier for the object.
            
        Raises:
            RuntimeError: If video has not been initialized.
        """
        if self.inference_state is None:
             raise RuntimeError("Video not initialized.")

        _, _, _ = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )

    def propagate(self) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Propagates the masks through the entire video using the memory.
        
        Consumes the generator from `propagate_in_video` to return the full track.
        
        Returns:
            Dict[int, Dict[int, np.ndarray]]: A dictionary mapping frame_idx -> {obj_id -> binary_mask}.
            
        Raises:
            RuntimeError: If video has not been initialized.
        """
        if self.inference_state is None:
             raise RuntimeError("Video not initialized.")
        
        results = {}
        
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
             frame_res = {}
             for i, obj_id in enumerate(out_obj_ids):
                 # Threshold logits to get binary mask
                 mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                 frame_res[obj_id] = mask
             results[out_frame_idx] = frame_res
             
        return results

    def reset_state(self):
        """
        Resets inference state but keeps video loaded if possible, or just clears all.
        """
        if self.inference_state:
            self.predictor.reset_state(self.inference_state)

    def cleanup(self):
        """
        Cleans up temporary resources.
        """
        if self.video_dir_ctx:
            self.video_dir_ctx.cleanup()
            self.video_dir_ctx = None
