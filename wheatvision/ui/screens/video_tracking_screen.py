import gradio as gr
import numpy as np
import cv2
import os
import json
from typing import List, Dict, Tuple, Optional
from wheatvision.integrations.sam2_adapter import Sam2Adapter, Sam2NotAvailable
from wheatvision.integrations.sam2_adapter.video_utils import extract_frames, find_sharpest_frame_idx

# State Classes to manage session data
class VideoTrackingState:
    def __init__(self):
        self.frames: List[np.ndarray] = []
        self.original_fps: float = 30.0
        self.video_path: str = ""
        self.tracker_service = None # VideoTrackingService
        self.init_frame_idx: int = 0
        self.masks_per_frame: Dict[int, Dict[int, np.ndarray]] = {} # frame_idx -> {obj_id -> mask}
        self.proposals: List[Dict] = [] # Auto-generated proposals on init frame
        self.frame_previews: Dict[int, np.ndarray] = {} # Cached display images
        self.obj_colors: Dict[int, Tuple[int, int, int]] = {}
    
    def cleanup(self):
        if self.tracker_service:
            self.tracker_service.cleanup()

def _get_overlay_image(image: np.ndarray, masks: Dict[int, np.ndarray], colors: Dict[int, Tuple]) -> np.ndarray:
    """
    Overlays masks on the image with transparency and contours.
    
    Args:
        image (np.ndarray): The background image.
        masks (Dict[int, np.ndarray]): Dictionary mapping object IDs to binary masks.
        colors (Dict[int, Tuple]): Dictionary mapping object IDs to RGB color tuples (or BGR depending on context).
        
    Returns:
        np.ndarray: Image with overlaid masks.
    """
    vis = image.copy()
    if not masks:
        return vis
        
    # Create overlay
    overlay = np.zeros_like(vis, dtype=np.uint8)
    alpha_mask = np.zeros(vis.shape[:2], dtype=np.float32)
    
    for obj_id, mask in masks.items():
        if obj_id not in colors:
            colors[obj_id] = tuple(np.random.randint(0, 255, 3).tolist())
        color = colors[obj_id]
        
        # Draw mask
        m = mask.astype(bool)
        overlay[m] = color
        alpha_mask[m] = 0.5 # Transparency
        
        # Draw contour
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
        
    # Blend
    alpha_mask = alpha_mask[..., None]
    vis = (vis * (1 - alpha_mask) + overlay * alpha_mask).astype(np.uint8)
    return vis

def build_video_tracking_tab():
    gr.Markdown("## Video Tracking (SAM2 X-Frame)")
    
    state = gr.State(VideoTrackingState())
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video", sources=["upload"])
            analyze_btn = gr.Button("1. Analyze & Load Video", variant="primary")
            
            gr.Markdown("### Initialization Settings")
            init_strategy = gr.Radio(
                ["Auto-Sharpest", "First Frame", "Manual Frame"], 
                value="Auto-Sharpest", 
                label="Frame Selection Strategy"
            )
            manual_frame_idx = gr.Number(value=0, label="Manual Frame Index", precision=0, visible=False)
            
            with gr.Accordion("Advanced Init Grid Settings", open=False):
                points_per_side = gr.Slider(16, 64, value=32, step=4, label="Points per Side")
                pred_iou = gr.Slider(0.5, 1.0, value=0.85, label="Pred IOU Threshold")
                stab_score = gr.Slider(0.5, 1.0, value=0.90, label="Stability Score Threshold")
            
            propose_btn = gr.Button("2. Auto-Propose Candidates", variant="secondary")
            
        with gr.Column(scale=2):
            gallery = gr.Gallery(label="Candidate Proposals", columns=4, height=300, allow_preview=True)
            gr.Markdown("select candidates from above to track")
            track_selected_btn = gr.Button("3. Initialize Tracking with Selected", variant="primary")
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=3):
            main_display = gr.Image(label="Tracking View", interactive=True)
            scrubber = gr.Slider(0, 100, value=0, label="Frame Scrubber", step=1)
            
        with gr.Column(scale=1):
            correction_mode = gr.Radio(["View", "Positive Click", "Negative Click"], value="View", label="Correction Tool")
            propagate_btn = gr.Button("4. Propagate / Update Tracking", variant="primary")
            reset_btn = gr.Button("Reset Tracking", variant="stop")
            
            export_btn = gr.Button("5. Export Results")
            export_file = gr.File(label="Download Export")

    # --- Event Handlers ---

    def analyze_video(video_file, st: VideoTrackingState):
        if video_file is None:
            return st, gr.update(), gr.update(maximum=0, value=0)
        
        # Reset state
        if st.tracker_service:
            st.tracker_service.cleanup()
        st = VideoTrackingState()
        st.video_path = video_file
        
        # Load Frames (Limit to e.g. 300 for memory safety efficiently, or just load all if small)
        # For this implementation we load all but handle with care.
        st.frames = extract_frames(video_file)
        if not st.frames:
            gr.Warning("No frames could be extracted.")
            return st, None, gr.update()
            
        # Init Tracker Service
        try:
            adapter = Sam2Adapter()
            st.tracker_service = adapter.get_video_tracker()
            st.tracker_service.init_video(st.frames)
        except Sam2NotAvailable as e:
            gr.Warning(f"SAM2 not available: {e}")
            return st, None, gr.update()
        except Exception as e:
            gr.Warning(f"Failed to init tracker: {e}")
            return st, None, gr.update()
            
        gr.Info(f"Loaded {len(st.frames)} frames.")
        return st, st.frames[0], gr.update(maximum=len(st.frames)-1, value=0)

    def update_frame_selector_visibility(choice):
        return gr.update(visible=(choice == "Manual Frame"))

    def auto_propose(st: VideoTrackingState, strategy, manual_idx, pps, iou, stab):
        if not st.frames:
            gr.Warning("Load video first.")
            return st, []
            
        # Determine Init Frame
        if strategy == "Auto-Sharpest":
            tgt_idx = find_sharpest_frame_idx(st.frames[:30]) # Check first 30 frames for speed
            if tgt_idx == -1: tgt_idx = 0
        elif strategy == "Manual Frame":
            tgt_idx = int(manual_idx)
        else:
            tgt_idx = 0
            
        tgt_idx = min(max(0, tgt_idx), len(st.frames)-1)
        st.init_frame_idx = tgt_idx
        
        target_img = st.frames[tgt_idx]
        adapter = Sam2Adapter()
        adapter.build() # Ensure predictor is built
        
        label_map = adapter.oversegment(
            image_bgr=target_img,
            points_per_side=int(pps),
            predicted_intersection_over_union_threshold=float(iou),
            stability_score_threshold=float(stab),
            # Defaults for others
            box_non_maximum_suppression_threshold=0.7,
            crop_layer_count=0,
            crop_overlap_ratio=0,
            minimum_mask_region_area=100,
            points_per_batch=64
        )
        
        # Convert label map to individual masks for gallery
        unique_ids = np.unique(label_map)
        unique_ids = unique_ids[unique_ids > 0] # skip bg
        
        gallery_imgs = []
        st.proposals = []
        
        # Limit to top 20 to avoid UI flood
        count = 0
        for uid in unique_ids:
            if count > 20: break
            mask = (label_map == uid)
            
            # Crop to mask for gallery thumbnail
            y, x = np.where(mask)
            if len(y) == 0: continue
            y1, y2, x1, x2 = y.min(), y.max(), x.min(), x.max()
            
            # Add padding
            h, w = target_img.shape[:2]
            pad = 10
            y1, y2 = max(0, y1-pad), min(h, y2+pad)
            x1, x2 = max(0, x1-pad), min(w, x2+pad)
            
            thumb = target_img[y1:y2, x1:x2].copy()
             # Draw mask on thumb
            mask_crop = mask[y1:y2, x1:x2]
            thumb[mask_crop] = (thumb[mask_crop] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
            
            gallery_imgs.append((thumb, f"ID {uid}"))
            st.proposals.append({"id": uid, "mask": mask})
            count += 1
            
        gr.Info(f"Found {len(unique_ids)} candidates. Showing top {len(gallery_imgs)}.")
        
        # Show whole frame with overlay in main view
        overlay_init = _get_overlay_image(target_img, {p["id"]: p["mask"] for p in st.proposals}, {})
        return st, gallery_imgs, overlay_init, gr.update(value=tgt_idx)

    def init_tracking(st: VideoTrackingState, evt: gr.SelectData):
        if not st.proposals:
            return st
        
        # User selected an index from gallery
        selected_idx = evt.index
        if selected_idx >= len(st.proposals):
            return st
            
        prop = st.proposals[selected_idx]
        mask = prop["mask"]
        obj_id = prop["id"] # Use the ID from proposal as obj_id
        
        # Init Tracker
        try:
            # We assume user wants to ADD to current tracking, or start fresh?
            # Let's support single object tracking for now for simplicity, or multi if they click multiple times?
            # For this 'init' button, let's treat it as "Add this object to tracker"
            
            st.tracker_service.add_mask(st.init_frame_idx, mask, obj_id)
            
            # Store in state
            if st.init_frame_idx not in st.masks_per_frame:
                st.masks_per_frame[st.init_frame_idx] = {}
            st.masks_per_frame[st.init_frame_idx][obj_id] = mask
            
            gr.Info(f"Added ID {obj_id} to tracking.")
        except Exception as e:
            gr.Warning(f"Error adding mask: {e}")
            
        return st

    def propagate(st: VideoTrackingState):
        if not st.tracker_service:
            return st, gr.update()
        
        try:
            results = st.tracker_service.propagate()
            # results is frame_idx -> {obj_id -> mask}
            st.masks_per_frame = results
            gr.Info("Propagation complete.")
        except Exception as e:
            gr.Warning(f"Propagation failed: {e}")
            
        # Update current view
        return st

    def on_scrub(val, st: VideoTrackingState):
        if not st.frames: return None
        idx = int(val)
        idx = min(idx, len(st.frames)-1)
        
        img = st.frames[idx]
        if idx in st.masks_per_frame:
            masks = st.masks_per_frame[idx]
            vis = _get_overlay_image(img, masks, st.obj_colors)
        else:
            vis = img
            
        return vis

    def handle_click(st, evt: gr.SelectData, frame_idx, mode):
        if mode == "View" or not st.tracker_service:
            return st, gr.update()
            
        points = np.array([[evt.index[0], evt.index[1]]], dtype=np.float32)
        labels = np.array([1 if mode == "Positive Click" else 0], dtype=np.int32)
        
        # For simplicity, assume we are correcting Object ID 1 if multiple exists, or the last added?
        # A robust UI would allow selecting WHICH object to correct.
        # Let's assume ID 1 or the first available ID in that frame, or create new?
        # To keep it simple: We correct the object that is CLOSEST to the click?
        # Or just default to ID 1.
        
        frame_data = st.masks_per_frame.get(frame_idx, {})
        target_obj_id = 1
        if frame_data:
             target_obj_id = list(frame_data.keys())[0] # Pick first
        
        try:
            st.tracker_service.add_click(frame_idx, points, labels, target_obj_id)
            gr.Info(f"Added click to frame {frame_idx} for ID {target_obj_id}")
            
            # Immediately re-propagate? Or wait for button?
            # Usually better to wait for 'Propagate' button to avoid heavy compute on every click
        except Exception as e:
            gr.Warning(f"Correction failed: {e}")
            
        return st

    def export_video(st: VideoTrackingState):
        if not st.masks_per_frame:
            return None
        
        try:
            # Create MP4
            h, w = st.frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = "export_tracking.mp4"
            out = cv2.VideoWriter(out_path, fourcc, 10.0, (w, h))
            
            for i, frame in enumerate(st.frames):
                masks = st.masks_per_frame.get(i, {})
                vis = _get_overlay_image(frame, masks, st.obj_colors)
                out.write(vis)
            out.release()
            
            return out_path
        except Exception as e:
            gr.Warning(f"Export failed: {e}")
            return None

    # Wiring
    analyze_btn.click(analyze_video, inputs=[video_input, state], outputs=[state, main_display, scrubber])
    init_strategy.change(update_frame_selector_visibility, inputs=init_strategy, outputs=manual_frame_idx)
    
    propose_btn.click(
        auto_propose, 
        inputs=[state, init_strategy, manual_frame_idx, points_per_side, pred_iou, stab_score],
        outputs=[state, gallery, main_display, scrubber]
    )
    
    gallery.select(init_tracking, inputs=[state], outputs=[state])
    
    propagate_btn.click(propagate, inputs=[state], outputs=[state]).then(
        on_scrub, inputs=[scrubber, state], outputs=[main_display]
    )
    
    scrubber.change(on_scrub, inputs=[scrubber, state], outputs=[main_display])
    
    main_display.select(
        handle_click,
        inputs=[state, scrubber, correction_mode],
        outputs=[state]
    )
    
    export_btn.click(export_video, inputs=[state], outputs=[export_file])

    return
