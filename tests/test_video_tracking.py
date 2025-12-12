import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

# Mock sam2 module before import
sys.modules["sam2"] = MagicMock()
sys.modules["sam2.build_sam2_video_predictor"] = MagicMock()

from wheatvision.integrations.sam2_adapter.video_tracking_service import VideoTrackingService
import torch

class TestVideoTrackingService(unittest.TestCase):
    def setUp(self):
        self.mock_predictor = MagicMock()
        self.mock_inference_state = "mock_state"
        self.mock_predictor.init_state.return_value = self.mock_inference_state
        
        # Mock propagate to yield one frame
        # yield (frame_idx, obj_ids, mask_logits)
        # mask_logits shape: (num_objs, 1, H, W)
        self.mock_predictor.propagate_in_video.return_value = [
            (0, [1], [torch.tensor([[[[1.0]]]])]) 
        ]
        # Expectation for add methods to return 3 values
        self.mock_predictor.add_new_mask.return_value = (None, None, None)
        self.mock_predictor.add_new_points_or_box.return_value = (None, None, None)
        
        # Setup the build mock
        self.patcher = patch("wheatvision.integrations.sam2_adapter.video_tracking_service.build_sam2_video_predictor", return_value=self.mock_predictor)
        self.mock_build = self.patcher.start()
        
        # We also need to mock os.path.exists to pass check
        self.path_patcher = patch("os.path.exists", return_value=True)
        self.mock_exists = self.path_patcher.start()
        
        # Mock torch for the propagate yield logic if it uses torch
        
    def tearDown(self):
        self.patcher.stop()
        self.path_patcher.stop()

    def test_init_flow(self):
        service = VideoTrackingService("config.yaml", "ckpt.pt")
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        
        service.init_video(frames)
        
        self.mock_build.assert_called_once()
        self.mock_predictor.init_state.assert_called_once()
        self.assertEqual(service.frames_count, 1)

    def test_add_mask_and_propagate(self):
        service = VideoTrackingService("config.yaml", "ckpt.pt")
        frames = [np.zeros((10, 10, 3), dtype=np.uint8)]
        service.init_video(frames)
        
        mask = np.zeros((10, 10), dtype=np.uint8)
        service.add_mask(0, mask, 1)
        self.mock_predictor.add_new_mask.assert_called()
        
        # Fix mock yield for propagate
        # The service expects torch tensors in output
        mock_logits = torch.randn(1, 1, 10, 10)
        self.mock_predictor.propagate_in_video.return_value = iter([
            (0, [1], mock_logits)
        ])
        
        results = service.propagate()
        self.assertIn(0, results)
        self.assertIn(1, results[0])
        self.assertEqual(results[0][1].shape, (10, 10))

if __name__ == "__main__":
    unittest.main()
