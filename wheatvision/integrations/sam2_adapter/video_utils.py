import cv2
import numpy as np
from typing import List, Tuple, Generator

def extract_frames_generator(video_path: str, max_frames: int = None, stride: int = 1) -> Generator[np.ndarray, None, None]:
    """
    Yields frames from a video file.
    
    Args:
        video_path (str): Path to the video file.
        max_frames (int, optional): Maximum number of frames to yield. If None, yields all.
        stride (int): Yield every Nth frame.
        
    Yields:
        np.ndarray: Video frame in BGR format.
        
    Raises:
        IOError: If could not open video at video_path.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video at {video_path}")
    
    count = 0
    yielded_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if count % stride == 0:
                yield frame
                yielded_count += 1
                if max_frames is not None and yielded_count >= max_frames:
                    break
            count += 1
    finally:
        cap.release()

def extract_frames(video_path: str, max_frames: int = None, stride: int = 1) -> List[np.ndarray]:
    """
    Extracts frames from a video file into a list.
    
    Args:
        video_path (str): Path to the video file.
        max_frames (int, optional): Maximum number of frames to extract.
        stride (int): Extract every Nth frame.
        
    Returns:
        List[np.ndarray]: List of extracted frames.
    """
    return list(extract_frames_generator(video_path, max_frames, stride))

def compute_frame_sharpness(image: np.ndarray) -> float:
    """
    Computes a sharpness score for an image using the Variance of Laplacian method.
    
    Higher score indicates a sharper image.
    
    Args:
        image (np.ndarray): Input image.
        
    Returns:
        float: Sharpness score.
    """
    if image is None:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def find_sharpest_frame_idx(frames: List[np.ndarray]) -> int:
    """
    Returns the index of the sharpest frame in the list.
    
    Args:
        frames (List[np.ndarray]): List of frames to analyze.
        
    Returns:
        int: Index of the sharpest frame, or -1 if list is empty.
    """
    if not frames:
        return -1
    scores = [compute_frame_sharpness(f) for f in frames]
    return int(np.argmax(scores))
