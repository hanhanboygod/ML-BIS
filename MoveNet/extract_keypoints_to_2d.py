"""
MoveNet Keypoint Extraction to 2D Image for CNN Training

SETTINGS SUMMARY:
1. Video format: MP4
2. Multiple people handling: MultiPose model (detects up to 6 people per frame)
3. Temporal representation: 
   - Each frame becomes a column in the output 2D array
   - Normalized to fixed number of frames (default: 64 frames)
4. Keypoint features: 17 keypoints × 3 values (y, x, confidence) = 51 features per frame
5. Output format: NumPy array (.npy) - most effective for 2D CNN training
6. Model: MoveNet MultiPose Lightning (detects multiple people)

ANNOTATION PROCESSING (Training):
- Parses CVAT XML annotation files to extract tracks (clips)
- Processes FULL FRAMES (no cropping) with MultiPose to detect all people
- Uses annotation bounding boxes to MATCH and identify the correct person among detections
- Matching strategy: IoU (Intersection over Union) between detection bbox and annotation bbox
- Processes all tracks in the annotation file
- Output naming: {video_name}_{label}_{track_id}_keypoints.npy
- Only processes frames where bounding box exists and outside="0" (person visible)
- Video files should be in the same directory as annotation XML files

INFERENCE WORKFLOW (After Training):
- Run MultiPose on new video frames (no annotations needed)
- Detect all people in frame
- Extract keypoints for each detected person
- Feed each person's keypoints to trained CNN classifier
- Get classification for each person independently

Output shape: (51, num_frames) where:
- Rows: 51 features (17 keypoints × 3 coordinates each: y, x, confidence)
- Cols: Normalized number of frames (default: 64)

Generated partially by Github Copilot
Date: 2025-10-25
"""

import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import xml.etree.ElementTree as ET
import argparse


# Configure GPU settings
def configure_gpu():
    """Configure TensorFlow to use GPU efficiently."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"✓ GPU acceleration enabled: {len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            return True
        except RuntimeError as e:
            print(f"⚠ GPU configuration error: {e}")
            return False
    else:
        print("⚠ No GPU detected. Running on CPU.")
        return False


# Configure GPU before loading models
GPU_AVAILABLE = configure_gpu()
print("-" * 60)


class CVATAnnotationParser:
    """Parse CVAT XML annotation files to extract track information."""
    
    def __init__(self, xml_path: str):
        """
        Initialize parser with XML file path.
        
        Args:
            xml_path: Path to CVAT XML annotation file
        """
        self.xml_path = xml_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
    
    def get_video_info(self) -> Dict[str, any]:
        """Extract video metadata from annotation file."""
        meta = self.root.find('meta')
        if meta is not None:
            original_size = meta.find('original_size')
            if original_size is not None:
                width = int(original_size.find('width').text)
                height = int(original_size.find('height').text)
                return {'width': width, 'height': height}
        return {}
    
    def get_tracks(self) -> List[Dict]:
        """
        Extract all tracks from the annotation file.
        
        Returns:
            List of track dictionaries with track info and bounding boxes
        """
        tracks = []
        
        for track in self.root.findall('track'):
            track_id = track.get('id')
            label = track.get('label')
            
            # Extract all boxes for this track
            boxes = []
            for box in track.findall('box'):
                frame_num = int(box.get('frame'))
                outside = int(box.get('outside', '0'))
                
                # Only include frames where person is visible (outside="0")
                if outside == 0:
                    box_data = {
                        'frame': frame_num,
                        'xtl': float(box.get('xtl')),
                        'ytl': float(box.get('ytl')),
                        'xbr': float(box.get('xbr')),
                        'ybr': float(box.get('ybr')),
                        'occluded': int(box.get('occluded', '0'))
                    }
                    
                    # Extract attributes if present
                    attributes = {}
                    for attr in box.findall('attribute'):
                        attr_name = attr.get('name')
                        attributes[attr_name] = attr.text
                    box_data['attributes'] = attributes
                    
                    boxes.append(box_data)
            
            if boxes:  # Only add tracks that have visible frames
                tracks.append({
                    'id': track_id,
                    'label': label,
                    'boxes': boxes
                })
        
        return tracks


class MoveNetKeypointExtractor:
    """Extract pose keypoints from videos using MoveNet and convert to 2D arrays."""
    
    # MoveNet outputs 17 keypoints
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def __init__(self, model_name: str = 'movenet_multipose_lightning'):
        """
        Initialize the MoveNet model.
        
        Args:
            model_name: Model to use. Options:
                - 'movenet_singlepose_lightning' (faster, single person)
                - 'movenet_singlepose_thunder' (slower, more accurate, single person)
                - 'movenet_multipose_lightning' (detects up to 6 people)
        """
        print(f"Loading {model_name} model...")
        
        model_url_map = {
            'movenet_singlepose_lightning': 
                'https://tfhub.dev/google/movenet/singlepose/lightning/4',
            'movenet_singlepose_thunder': 
                'https://tfhub.dev/google/movenet/singlepose/thunder/4',
            'movenet_multipose_lightning':
                'https://tfhub.dev/google/movenet/multipose/lightning/1'
        }
        
        if model_name not in model_url_map:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model_name = model_name
        self.is_multipose = 'multipose' in model_name
        
        device = "GPU" if GPU_AVAILABLE else "CPU"
        self.model = hub.load(model_url_map[model_name])
        self.movenet = self.model.signatures['serving_default']
        
        # Input size for the model
        if 'lightning' in model_name:
            self.input_size = 256 if self.is_multipose else 192
        else:
            self.input_size = 256
        
        print(f"✓ Model loaded successfully on {device}! Input size: {self.input_size}x{self.input_size}")
        print(f"  MultiPose mode: {self.is_multipose}")
    
    def calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: Dict with keys 'xtl', 'ytl', 'xbr', 'ybr'
            box2: Dict with keys 'xtl', 'ytl', 'xbr', 'ybr'
            
        Returns:
            IoU score (0 to 1)
        """
        # Calculate intersection
        x_left = max(box1['xtl'], box2['xtl'])
        y_top = max(box1['ytl'], box2['ytl'])
        x_right = min(box1['xbr'], box2['xbr'])
        y_bottom = min(box1['ybr'], box2['ybr'])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        box1_area = (box1['xbr'] - box1['xtl']) * (box1['ybr'] - box1['ytl'])
        box2_area = (box2['xbr'] - box2['xtl']) * (box2['ybr'] - box2['ytl'])
        union_area = box1_area + box2_area - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def keypoints_to_bbox(self, keypoints: np.ndarray, width: int, height: int) -> Dict:
        """
        Calculate bounding box from keypoints.
        
        Args:
            keypoints: Array of shape (17, 3) with [y, x, confidence]
            width: Frame width
            height: Frame height
            
        Returns:
            Dict with 'xtl', 'ytl', 'xbr', 'ybr'
        """
        # Filter out low confidence keypoints
        confident_kp = keypoints[keypoints[:, 2] > 0.3]
        
        if len(confident_kp) == 0:
            # Return empty box if no confident keypoints
            return {'xtl': 0, 'ytl': 0, 'xbr': 0, 'ybr': 0}
        
        # Convert normalized coordinates to pixel coordinates
        y_coords = confident_kp[:, 0] * height
        x_coords = confident_kp[:, 1] * width
        
        # Add some padding
        padding = 20
        xtl = max(0, np.min(x_coords) - padding)
        ytl = max(0, np.min(y_coords) - padding)
        xbr = min(width, np.max(x_coords) + padding)
        ybr = min(height, np.max(y_coords) + padding)
        
        return {'xtl': xtl, 'ytl': ytl, 'xbr': xbr, 'ybr': ybr}
    
    def extract_keypoints_from_frame(
        self, 
        frame: np.ndarray,
        annotation_bbox: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Extract keypoints from a single frame.
        
        Args:
            frame: BGR image from OpenCV (H, W, 3)
            annotation_bbox: Optional annotation bounding box to match against
                           (Dict with 'xtl', 'ytl', 'xbr', 'ybr')
            
        Returns:
            Keypoints array of shape (17, 3) containing [y, x, confidence]
            For MultiPose: returns keypoints of best matching person
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        # Resize and pad to maintain aspect ratio
        img = tf.image.resize_with_pad(
            tf.expand_dims(frame_rgb, axis=0),
            self.input_size,
            self.input_size
        )
        
        # Convert to int32
        input_image = tf.cast(img, dtype=tf.int32)
        
        # Run model inference (automatically uses GPU if available)
        outputs = self.movenet(input_image)
        
        if self.is_multipose:
            # MultiPose output shape: (1, 6, 56)
            # 6 people max, 56 values = 17 keypoints * 3 (y, x, score) + 4 bbox + 1 score
            keypoints_with_scores = outputs['output_0'].numpy()[0]
            
            if annotation_bbox is None:
                # No annotation bbox provided, return highest scoring detection
                detection_scores = keypoints_with_scores[:, 55]  # Last value is detection score
                best_idx = np.argmax(detection_scores)
                keypoints = keypoints_with_scores[best_idx, :51].reshape(17, 3)
            else:
                # Match detections to annotation bbox using IoU
                best_iou = 0.0
                best_keypoints = None
                
                for detection in keypoints_with_scores:
                    # Extract keypoints (first 51 values)
                    kp = detection[:51].reshape(17, 3)
                    
                    # Calculate bbox from keypoints
                    detection_bbox = self.keypoints_to_bbox(kp, width, height)
                    
                    # Calculate IoU with annotation bbox
                    iou = self.calculate_iou(annotation_bbox, detection_bbox)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_keypoints = kp
                
                # If no good match found (IoU < 0.1), use highest scoring detection
                if best_iou < 0.1:
                    detection_scores = keypoints_with_scores[:, 55]
                    best_idx = np.argmax(detection_scores)
                    keypoints = keypoints_with_scores[best_idx, :51].reshape(17, 3)
                else:
                    keypoints = best_keypoints
        else:
            # SinglePose output shape: (1, 1, 17, 3)
            keypoints = outputs['output_0'].numpy().squeeze()
        
        return keypoints
    
    def process_video(
        self,
        video_path: str,
        target_frames: int = 64,
        frame_skip: int = 1
    ) -> Tuple[np.ndarray, dict]:
        """
        Process a video and extract keypoints for all frames.
        
        Args:
            video_path: Path to the video file
            target_frames: Number of frames to normalize to
            frame_skip: Process every Nth frame (1 = process all frames)
            
        Returns:
            Tuple of:
                - 2D array of shape (51, target_frames) containing keypoints
                - Metadata dictionary with video information
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nProcessing: {os.path.basename(video_path)}")
        print(f"  Total frames: {total_frames}, FPS: {fps:.2f}, Size: {width}x{height}")
        
        # Collect keypoints from all frames
        all_keypoints = []
        frame_idx = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if needed
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            # Extract keypoints
            keypoints = self.extract_keypoints_from_frame(frame)
            all_keypoints.append(keypoints)
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"  Processed {processed_count} frames...", end='\r')
            
            frame_idx += 1
        
        cap.release()
        print(f"  Processed {processed_count} frames total.    ")
        
        if len(all_keypoints) == 0:
            raise ValueError("No frames were processed from the video")
        
        # Convert to numpy array: (num_frames, 17, 3)
        keypoints_array = np.array(all_keypoints)
        
        # Normalize temporal dimension to target_frames
        normalized_keypoints = self._normalize_temporal_dimension(
            keypoints_array, target_frames, fps=fps
        )
        
        # Reshape to 2D: (51, target_frames)
        # Each column is one frame, each row is a feature
        keypoints_2d = normalized_keypoints.reshape(target_frames, -1).T
        
        # Prepare metadata
        metadata = {
            'video_path': video_path,
            'video_name': os.path.basename(video_path),
            'original_frames': processed_count,
            'normalized_frames': target_frames,
            'fps': fps,
            'resolution': (width, height),
            'frame_skip': frame_skip,
            'output_shape': keypoints_2d.shape
        }
        
        return keypoints_2d, metadata
    
    def _normalize_temporal_dimension(
        self,
        keypoints: np.ndarray,
        target_frames: int,
        fps: float = 30.0,
        target_duration: float = 2.0
    ) -> np.ndarray:
        """
        Normalize temporal dimension by resampling to fixed duration (2s) -> fixed frames (64).
        Preserves action speed by padding or cropping.
        """
        num_frames = keypoints.shape[0]
        duration = num_frames / fps
        
        # Input time axis (relative to the start of the clip)
        t_in = np.arange(num_frames) / fps
        
        # Align the END of the input clip with the END of the target duration (2.0s)
        # If duration < 2.0s: t_in starts > 0. The beginning (0 to start) is padded with the first frame.
        # If duration > 2.0s: t_in starts < 0. The beginning is cropped.
        start_offset = target_duration - duration
        t_in = t_in + start_offset
        
        # Output time axis (0 to 2.0s)
        t_out = np.linspace(0, target_duration, target_frames)
        
        # Interpolate
        resampled = np.zeros((target_frames, 17, 3))
        for kp_idx in range(17):
            for coord_idx in range(3):
                resampled[:, kp_idx, coord_idx] = np.interp(
                    t_out,
                    t_in,
                    keypoints[:, kp_idx, coord_idx]
                )
                
        return resampled
    
    def process_track(
        self,
        video_path: str,
        track_info: Dict,
        target_frames: int = 64
    ) -> Tuple[np.ndarray, dict]:
        """
        Process a single track from annotation file.
        
        Args:
            video_path: Path to the video file
            track_info: Dictionary containing track information (from CVATAnnotationParser)
            target_frames: Number of frames to normalize to
            
        Returns:
            Tuple of:
                - 2D array of shape (51, target_frames) containing keypoints
                - Metadata dictionary with track information
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        track_id = track_info['id']
        label = track_info['label']
        boxes = track_info['boxes']
        
        print(f"\nProcessing Track {track_id} ({label}): {os.path.basename(video_path)}")
        print(f"  Total annotated frames: {len(boxes)}")
        print(f"  Frame range: {boxes[0]['frame']} to {boxes[-1]['frame']}")
        
        # Collect keypoints from annotated frames
        all_keypoints = []
        
        # --- Pre-roll Context Extraction ---
        # If action is shorter than 2.0s, try to extract previous frames from video
        # to provide correct "idle/pre-action" context for the model
        annotated_duration = len(boxes) / fps
        target_duration = 2.0
        
        if annotated_duration < target_duration:
            needed_seconds = target_duration - annotated_duration
            needed_frames = int(np.ceil(needed_seconds * fps))
            
            start_frame = int(boxes[0]['frame'])
            # We can go back at most to frame 0
            pre_start_frame = max(0, start_frame - needed_frames)
            
            if pre_start_frame < start_frame:
                print(f"  Extracting pre-roll context: frames {pre_start_frame} to {start_frame-1}")
                
                # Use the first annotated frame's bbox as reference to find the person
                ref_box = boxes[0]
                ref_bbox = {
                    'xtl': float(ref_box['xtl']),
                    'ytl': float(ref_box['ytl']),
                    'xbr': float(ref_box['xbr']),
                    'ybr': float(ref_box['ybr'])
                }
                
                for frame_idx in range(pre_start_frame, start_frame):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                        
                    # Extract using reference bbox
                    kp = self.extract_keypoints_from_frame(frame, ref_bbox)
                    all_keypoints.append(kp)
                    if frame_idx % 10 == 0:
                        print(f"  Processed pre-roll frame {frame_idx}...", end='\r')
                print(f"  Extracted {len(all_keypoints)} pre-roll frames.        ")

        processed_count = 0
        
        for box_data in boxes:
            frame_num = box_data['frame']
            
            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                print(f"  Warning: Could not read frame {frame_num}, skipping...")
                continue
            
            # Prepare annotation bounding box for matching
            annotation_bbox = {
                'xtl': float(box_data['xtl']),
                'ytl': float(box_data['ytl']),
                'xbr': float(box_data['xbr']),
                'ybr': float(box_data['ybr'])
            }
            
            # Extract keypoints from FULL frame, matching to annotation bbox
            keypoints = self.extract_keypoints_from_frame(frame, annotation_bbox)
            all_keypoints.append(keypoints)
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"  Processed {processed_count}/{len(boxes)} annotated frames...", end='\r')
        
        cap.release()
        print(f"  Total frames collected: {len(all_keypoints)} ({len(all_keypoints) - processed_count} pre-roll + {processed_count} annotated)")
        
        if len(all_keypoints) == 0:
            raise ValueError(f"No frames were successfully processed for track {track_id}")
        
        # Convert to numpy array: (num_frames, 17, 3)
        keypoints_array = np.array(all_keypoints)
        
        # Normalize temporal dimension to target_frames
        normalized_keypoints = self._normalize_temporal_dimension(
            keypoints_array, target_frames, fps=fps
        )
        
        # Reshape to 2D: (51, target_frames)
        keypoints_2d = normalized_keypoints.reshape(target_frames, -1).T
        
        # Prepare metadata
        metadata = {
            'video_path': video_path,
            'video_name': os.path.basename(video_path),
            'track_id': track_id,
            'label': label,
            'original_frames': processed_count,
            'normalized_frames': target_frames,
            'fps': fps,
            'resolution': (width, height),
            'frame_range': (boxes[0]['frame'], boxes[-1]['frame']),
            'output_shape': keypoints_2d.shape
        }
        
        return keypoints_2d, metadata
    
    def process_video_with_annotations(
        self,
        annotation_path: str,
        output_dir: str,
        target_frames: int = 64
    ):
        """
        Process all tracks from a CVAT annotation file.
        
        Args:
            annotation_path: Path to CVAT XML annotation file
            output_dir: Directory to save output .npy files
            target_frames: Number of frames to normalize to
        """
        # Parse annotation file
        parser = CVATAnnotationParser(annotation_path)
        tracks = parser.get_tracks()
        
        if len(tracks) == 0:
            print(f"No tracks found in {annotation_path}")
            return
        
        # Determine video file path
        annotation_dir = Path(annotation_path).parent
        
        # Try to extract video name from annotation filename
        # Expected format: annotations_{video_name}.xml
        annotation_filename = Path(annotation_path).stem
        if annotation_filename.startswith('annotations_'):
            video_name = annotation_filename.replace('annotations_', '', 1)
            video_path = annotation_dir / f"{video_name}.mp4"
        else:
            # Try to find video from first track's attributes
            first_track_boxes = tracks[0]['boxes']
            if first_track_boxes and 'attributes' in first_track_boxes[0]:
                video_id = first_track_boxes[0]['attributes'].get('video_id', '')
                if video_id:
                    video_path = annotation_dir / video_id
                else:
                    raise ValueError(f"Could not determine video file from {annotation_path}")
            else:
                raise ValueError(f"Could not determine video file from {annotation_path}")
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nFound {len(tracks)} track(s) in {Path(annotation_path).name}")
        print(f"Video: {video_path.name}")
        print("=" * 60)
        
        results = []
        video_stem = video_path.stem
        
        for i, track_info in enumerate(tracks, 1):
            print(f"\n[{i}/{len(tracks)}]")
            
            try:
                # Process track
                keypoints_2d, metadata = self.process_track(
                    str(video_path),
                    track_info,
                    target_frames=target_frames
                )
                
                # Save as .npy file: {video_name}_{label}_{track_id}_keypoints.npy
                track_id = track_info['id']
                label = track_info['label']
                output_filename = f"{video_stem}_{label}_{track_id}_keypoints.npy"
                output_filepath = output_path / output_filename
                np.save(output_filepath, keypoints_2d)
                
                # Save metadata as well
                metadata_filename = f"{video_stem}_{label}_{track_id}_metadata.npy"
                metadata_filepath = output_path / metadata_filename
                np.save(metadata_filepath, metadata, allow_pickle=True)
                
                print(f"  Saved: {output_filename}")
                print(f"  Shape: {keypoints_2d.shape}")
                
                results.append({
                    'track_id': track_id,
                    'label': label,
                    'output': str(output_filepath),
                    'shape': keypoints_2d.shape,
                    'success': True
                })
                
            except Exception as e:
                print(f"  ERROR processing track {track_info['id']}: {str(e)}")
                results.append({
                    'track_id': track_info['id'],
                    'label': track_info['label'],
                    'success': False,
                    'error': str(e)
                })
        
        # Print summary
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        successful = sum(1 for r in results if r['success'])
        print(f"Total tracks: {len(tracks)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(tracks) - successful}")
        print(f"Output directory: {output_path}")
        
        # Verification step
        if successful > 0:
            print("\n" + "=" * 60)
            print("VERIFICATION")
            print("=" * 60)
            # Check the first successful output
            for res in results:
                if res['success']:
                    npy_path = res['output']
                    try:
                        data = np.load(npy_path)
                        print(f"Checking: {os.path.basename(npy_path)}")
                        print(f"  Shape: {data.shape} (Expected: (51, {target_frames}))")
                        
                        if data.shape != (51, target_frames):
                            print("  ❌ Shape mismatch!")
                        else:
                            print("  ✅ Shape correct.")
                            
                        # Check for NaN values
                        if np.isnan(data).any():
                            print("  ❌ Contains NaN values!")
                        else:
                            print("  ✅ No NaN values.")
                            
                        # Check value range (should be normalized 0-1 for coordinates)
                        # Note: Confidence scores (every 3rd value) are also 0-1
                        min_val = np.min(data)
                        max_val = np.max(data)
                        print(f"  Value range: [{min_val:.4f}, {max_val:.4f}]")
                        
                        if min_val < 0 or max_val > 1.0:
                             print("  ⚠️ Values outside [0, 1] range (might be okay if padding uses 0)")
                        else:
                             print("  ✅ Values within expected range.")
                             
                        break # Only check one file
                    except Exception as e:
                        print(f"  ❌ Error verifying file: {e}")
                    break
    
    def process_video_directory(
        self,
        input_dir: str,
        output_dir: str,
        target_frames: int = 64,
        frame_skip: int = 1,
        file_pattern: str = "*.mp4"
    ):
        """
        Process all videos in a directory.
        
        Args:
            input_dir: Directory containing video files
            output_dir: Directory to save output .npy files
            target_frames: Number of frames to normalize to
            frame_skip: Process every Nth frame
            file_pattern: Glob pattern for video files
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all video files
        video_files = list(input_path.glob(file_pattern))
        
        if len(video_files) == 0:
            print(f"No videos found matching '{file_pattern}' in {input_dir}")
            return
        
        print(f"\nFound {len(video_files)} video(s) to process")
        print("=" * 60)
        
        results = []
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}]")
            
            try:
                # Process video
                keypoints_2d, metadata = self.process_video(
                    str(video_file),
                    target_frames=target_frames,
                    frame_skip=frame_skip
                )
                
                # Save as .npy file
                output_filename = video_file.stem + '_keypoints.npy'
                output_filepath = output_path / output_filename
                np.save(output_filepath, keypoints_2d)
                
                # Save metadata as well
                metadata_filename = video_file.stem + '_metadata.npy'
                metadata_filepath = output_path / metadata_filename
                np.save(metadata_filepath, metadata, allow_pickle=True)
                
                print(f"  Saved: {output_filename}")
                print(f"  Shape: {keypoints_2d.shape}")
                
                results.append({
                    'video': str(video_file),
                    'output': str(output_filepath),
                    'shape': keypoints_2d.shape,
                    'success': True
                })
                
            except Exception as e:
                print(f"  ERROR processing {video_file.name}: {str(e)}")
                results.append({
                    'video': str(video_file),
                    'success': False,
                    'error': str(e)
                })
        
        # Print summary
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        successful = sum(1 for r in results if r['success'])
        print(f"Total videos: {len(video_files)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(video_files) - successful}")
        print(f"Output directory: {output_path}")
        

def main():
    parser = argparse.ArgumentParser(
        description='Extract MoveNet keypoints from videos and convert to 2D arrays for CNN training'
    )
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Processing mode')
    
    # Annotation mode parser
    annotation_parser = subparsers.add_parser(
        'annotation',
        help='Process video tracks from CVAT annotation file'
    )
    annotation_parser.add_argument(
        '--annotation',
        type=str,
        required=True,
        help='Path to CVAT XML annotation file'
    )
    annotation_parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory to save .npy files'
    )
    annotation_parser.add_argument(
        '--frames',
        type=int,
        default=64,
        help='Number of frames to normalize to (default: 64)'
    )
    annotation_parser.add_argument(
        '--model',
        type=str,
        default='movenet_multipose_lightning',
        choices=['movenet_singlepose_lightning', 'movenet_singlepose_thunder', 'movenet_multipose_lightning'],
        help='MoveNet model to use (default: multipose_lightning)'
    )
    annotation_parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='Force CPU usage even if GPU is available'
    )
    
    # Directory annotation mode parser
    dir_annotation_parser = subparsers.add_parser(
        'annotation-dir',
        help='Process all annotation files in a directory'
    )
    dir_annotation_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Directory containing CVAT XML annotation files'
    )
    dir_annotation_parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory to save .npy files'
    )
    dir_annotation_parser.add_argument(
        '--frames',
        type=int,
        default=64,
        help='Number of frames to normalize to (default: 64)'
    )
    dir_annotation_parser.add_argument(
        '--model',
        type=str,
        default='movenet_multipose_lightning',
        choices=['movenet_singlepose_lightning', 'movenet_singlepose_thunder', 'movenet_multipose_lightning'],
        help='MoveNet model to use (default: multipose_lightning)'
    )
    dir_annotation_parser.add_argument(
        '--pattern',
        type=str,
        default='annotations_*.xml',
        help='File pattern for annotation files (default: annotations_*.xml)'
    )
    dir_annotation_parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='Force CPU usage even if GPU is available'
    )
    
    # Video mode parser (original functionality)
    video_parser = subparsers.add_parser(
        'video',
        help='Process video file(s) directly without annotations'
    )
    video_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input video file or directory containing videos'
    )
    
    video_parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory to save .npy files'
    )
    
    video_parser.add_argument(
        '--frames',
        type=int,
        default=64,
        help='Number of frames to normalize to (default: 64)'
    )
    
    video_parser.add_argument(
        '--skip',
        type=int,
        default=1,
        help='Process every Nth frame (default: 1, process all frames)'
    )
    
    video_parser.add_argument(
        '--model',
        type=str,
        default='movenet_multipose_lightning',
        choices=['movenet_singlepose_lightning', 'movenet_singlepose_thunder', 'movenet_multipose_lightning'],
        help='MoveNet model to use (default: multipose_lightning)'
    )
    
    video_parser.add_argument(
        '--pattern',
        type=str,
        default='*.mp4',
        help='File pattern for videos (default: *.mp4)'
    )
    
    video_parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='Force CPU usage even if GPU is available'
    )
    
    args = parser.parse_args()
    
    # Override GPU setting if CPU-only flag is set
    if hasattr(args, 'cpu_only') and args.cpu_only:
        global GPU_AVAILABLE
        tf.config.set_visible_devices([], 'GPU')
        GPU_AVAILABLE = False
        print("⚠ Running in CPU-only mode (--cpu-only flag set)")
        print("-" * 60)
    
    if args.mode is None:
        parser.print_help()
        return 1
    
    # Initialize extractor
    extractor = MoveNetKeypointExtractor(model_name=args.model)
    
    if args.mode == 'annotation':
        # Process single annotation file
        print(f"\nProcessing annotation file: {args.annotation}")
        extractor.process_video_with_annotations(
            annotation_path=args.annotation,
            output_dir=args.output,
            target_frames=args.frames
        )
    
    elif args.mode == 'annotation-dir':
        # Process directory of annotation files
        input_path = Path(args.input)
        
        if not input_path.is_dir():
            print(f"Error: {args.input} is not a valid directory")
            return 1
        
        # Find all annotation files
        annotation_files = list(input_path.glob(args.pattern))
        
        if len(annotation_files) == 0:
            print(f"No annotation files found matching '{args.pattern}' in {args.input}")
            return 1
        
        print(f"\nFound {len(annotation_files)} annotation file(s)")
        print("=" * 60)
        
        for i, annotation_file in enumerate(annotation_files, 1):
            print(f"\n{'='*60}")
            print(f"Processing annotation file {i}/{len(annotation_files)}: {annotation_file.name}")
            print(f"{'='*60}")
            
            try:
                extractor.process_video_with_annotations(
                    annotation_path=str(annotation_file),
                    output_dir=args.output,
                    target_frames=args.frames
                )
            except Exception as e:
                print(f"ERROR processing {annotation_file.name}: {str(e)}")
    
    elif args.mode == 'video':
        # Original video processing mode
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Process single video
            print("\nProcessing single video...")
            keypoints_2d, metadata = extractor.process_video(
                str(input_path),
                target_frames=args.frames,
                frame_skip=args.skip
            )
            
            # Create output directory
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save output
            output_filename = input_path.stem + '_keypoints.npy'
            output_filepath = output_path / output_filename
            np.save(output_filepath, keypoints_2d)
            
            metadata_filename = input_path.stem + '_metadata.npy'
            metadata_filepath = output_path / metadata_filename
            np.save(metadata_filepath, metadata, allow_pickle=True)
            
            print(f"\nSaved to: {output_filepath}")
            print(f"Shape: {keypoints_2d.shape}")
            
        elif input_path.is_dir():
            # Process directory
            extractor.process_video_directory(
                input_dir=args.input,
                output_dir=args.output,
                target_frames=args.frames,
                frame_skip=args.skip,
                file_pattern=args.pattern
            )
        else:
            print(f"Error: {args.input} is not a valid file or directory")
            return 1
    
    print("\n✓ Done!")
    return 0


if __name__ == '__main__':
    exit(main())
