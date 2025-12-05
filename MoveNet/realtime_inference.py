"""
Real-time Pose Classification Inference

This script performs real-time inference on video streams using:
1. MoveNet MultiPose for keypoint detection
2. Your trained CNN model for action classification

Features:
- Detects multiple people simultaneously
- Maintains sliding window buffer for each person
- Classifies actions in real-time
- Displays results with bounding boxes and labels
- Supports webcam, video files, or video directories

Usage:
    # Webcam
    python realtime_inference.py --model model.h5 --labels labels.txt --source webcam
    
    # Video file
    python realtime_inference.py --model model.h5 --labels labels.txt --source video.mp4
    
    # Video directory
    python realtime_inference.py --model model.h5 --labels labels.txt --source videos/ --pattern "*.mp4"

Author: ML Final Project
Date: 2025-11-12
"""

import os
import sys
import argparse
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque
import time


def configure_gpu():
    """Configure TensorFlow to use GPU efficiently."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU acceleration enabled: {len(gpus)} GPU(s)")
            return True
        except RuntimeError as e:
            print(f"⚠ GPU configuration error: {e}")
            return False
    else:
        print("⚠ No GPU detected. Running on CPU.")
        return False


GPU_AVAILABLE = configure_gpu()
print("-" * 60)


class PersonTracker:
    """Track individual people across frames with time-based sliding window."""
    
    def __init__(self, person_id: int, initial_bbox: Dict, window_duration: float = 2.0):
        """
        Initialize person tracker.
        
        Args:
            person_id: Unique ID for this person
            initial_bbox: Initial bounding box {'xtl', 'ytl', 'xbr', 'ybr'}
            window_duration: Duration of sliding window in seconds (default: 2.0s)
        """
        self.id = person_id
        self.bbox = initial_bbox
        self.window_duration = window_duration
        # Buffer stores tuples of (timestamp, keypoints)
        self.keypoints_buffer = [] 
        self.last_seen_time = 0.0
        self.current_label = "detecting..."
        self.confidence = 0.0
        self.color = self._generate_color()
    
    def _generate_color(self):
        """Generate a unique color for this person."""
        np.random.seed(self.id * 42)
        return tuple(map(int, np.random.randint(50, 255, 3)))
    
    def update(self, bbox: Dict, keypoints: np.ndarray, timestamp: float):
        """Update tracker with new detection."""
        self.bbox = bbox
        self.keypoints_buffer.append((timestamp, keypoints))
        self.last_seen_time = timestamp
        
        # Remove old frames (keep only last window_duration seconds)
        cutoff_time = timestamp - self.window_duration
        while self.keypoints_buffer and self.keypoints_buffer[0][0] < cutoff_time:
            self.keypoints_buffer.pop(0)
    
    def is_active(self, current_time: float, max_missing_seconds: float = 1.0) -> bool:
        """Check if tracker is still active (recently seen)."""
        return (current_time - self.last_seen_time) <= max_missing_seconds
    
    def get_feature_window(self, target_frames: int = 64) -> Optional[np.ndarray]:
        """
        Get current feature window for classification, resampled to target_frames.
        
        Returns:
            Array of shape (target_frames, 17, 2) or None if not enough data
        """
        if len(self.keypoints_buffer) < 2:
            return None
            
        # Get timestamps and keypoints
        timestamps = np.array([t for t, _ in self.keypoints_buffer])
        keypoints = np.array([k for _, k in self.keypoints_buffer]) # (N, 17, 3)
        
        # Check if we have enough duration (e.g., at least 1/3 of the window)
        duration = timestamps[-1] - timestamps[0]
        if duration < (self.window_duration * 0.3):
            return None
            
        # Create target timestamps (uniformly spaced over the actual duration covered)
        # Note: We map the ACTUAL duration to 64 frames to preserve speed
        # If we mapped 2.0s to 64 frames but only had 1.0s of data, we'd get half a window.
        # Here we assume the model expects "filled" windows. 
        # Ideally, we want to resample the last 2.0s.
        
        current_time = timestamps[-1]
        start_time = current_time - self.window_duration
        
        # We need to interpolate the data within [start_time, current_time]
        # to exactly 'target_frames' points.
        target_times = np.linspace(start_time, current_time, target_frames)
        
        # Perform interpolation for each keypoint coordinate
        # keypoints shape: (N, 17, 3) -> we only need x,y (N, 17, 2)
        keypoints_xy = keypoints[:, :, :2]
        resampled_window = np.zeros((target_frames, 17, 2))
        
        for kp_idx in range(17):
            for coord_idx in range(2): # x and y
                resampled_window[:, kp_idx, coord_idx] = np.interp(
                    target_times,
                    timestamps,
                    keypoints_xy[:, kp_idx, coord_idx]
                )
                
        return resampled_window


class RealtimeClassifier:
    """Real-time pose classification with MoveNet + CNN."""
    
    def __init__(
        self,
        model_path: str,
        labels: List[str],
        window_size: int = 64,
        iou_threshold: float = 0.3,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize real-time classifier.
        
        Args:
            model_path: Path to trained CNN model (.h5, .keras, or SavedModel)
            labels: List of class labels
            window_size: Sliding window size (must match training)
            iou_threshold: IoU threshold for person matching
            confidence_threshold: Minimum confidence for classification
        """
        self.labels = labels
        self.window_size = window_size
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        
        # Load MoveNet
        print("Loading MoveNet MultiPose Lightning...")
        model_url = 'https://tfhub.dev/google/movenet/multipose/lightning/1'
        self.movenet_model = hub.load(model_url)
        self.movenet = self.movenet_model.signatures['serving_default']
        self.input_size = 256
        print("✓ MoveNet loaded successfully!")
        
        # Load CNN model
        print(f"Loading CNN model from {model_path}...")
        self.cnn_model = tf.keras.models.load_model(model_path)
        print(f"✓ CNN model loaded successfully!")
        print(f"  Input shape: {self.cnn_model.input_shape}")
        print(f"  Output classes: {len(labels)}")
        print(f"  Labels: {labels}")
        
        # Person tracking
        self.trackers: Dict[int, PersonTracker] = {}
        self.next_person_id = 0
        self.frame_count = 0
        
        # Performance metrics
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Frame skipping for FPS normalization
        self.target_fps = 30.0
        self.frame_accumulator = 0.0
    
    def calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """Calculate IoU between two bounding boxes."""
        x_left = max(box1['xtl'], box2['xtl'])
        y_top = max(box1['ytl'], box2['ytl'])
        x_right = min(box1['xbr'], box2['xbr'])
        y_bottom = min(box1['ybr'], box2['ybr'])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (box1['xbr'] - box1['xtl']) * (box1['ybr'] - box1['ytl'])
        area2 = (box2['xbr'] - box2['xtl']) * (box2['ybr'] - box2['ytl'])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def keypoints_to_bbox(self, keypoints: np.ndarray, width: int, height: int) -> Dict:
        """Calculate bounding box from keypoints."""
        confident_kp = keypoints[keypoints[:, 2] > 0.3]
        
        if len(confident_kp) == 0:
            return {'xtl': 0, 'ytl': 0, 'xbr': 0, 'ybr': 0}
        
        y_coords = confident_kp[:, 0] * height
        x_coords = confident_kp[:, 1] * width
        
        padding = 20
        xtl = max(0, np.min(x_coords) - padding)
        ytl = max(0, np.min(y_coords) - padding)
        xbr = min(width, np.max(x_coords) + padding)
        ybr = min(height, np.max(y_coords) + padding)
        
        return {'xtl': xtl, 'ytl': ytl, 'xbr': xbr, 'ybr': ybr}
    
    def detect_people(self, frame: np.ndarray) -> List[Tuple[Dict, np.ndarray]]:
        """
        Detect people in frame using MoveNet.
        
        Returns:
            List of (bbox, keypoints) tuples
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        # Resize for MoveNet
        img = tf.image.resize_with_pad(
            tf.expand_dims(frame_rgb, axis=0),
            self.input_size,
            self.input_size
        )
        input_image = tf.cast(img, dtype=tf.int32)
        
        # Run detection
        outputs = self.movenet(input_image)
        keypoints_with_scores = outputs['output_0'].numpy()[0]
        
        # Extract people with good confidence
        detections = []
        for detection in keypoints_with_scores:
            detection_score = detection[55]  # Overall detection score
            
            if detection_score > 0.1:  # Minimum detection threshold
                keypoints = detection[:51].reshape(17, 3)
                bbox = self.keypoints_to_bbox(keypoints, width, height)
                
                # Check if bbox is valid
                if bbox['xbr'] > bbox['xtl'] and bbox['ybr'] > bbox['ytl']:
                    detections.append((bbox, keypoints))
        
        return detections
    
    def match_detections_to_trackers(
        self,
        detections: List[Tuple[Dict, np.ndarray]],
        current_time: float
    ) -> Dict[int, Tuple[Dict, np.ndarray]]:
        """
        Match detections to existing trackers using IoU.
        
        Returns:
            Dict mapping tracker_id to (bbox, keypoints)
        """
        matches = {}
        used_detections = set()
        
        # Match each tracker to best detection
        for tracker_id, tracker in self.trackers.items():
            if not tracker.is_active(current_time):
                continue
            
            best_iou = 0.0
            best_detection_idx = -1
            
            for idx, (bbox, keypoints) in enumerate(detections):
                if idx in used_detections:
                    continue
                
                iou = self.calculate_iou(tracker.bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_detection_idx = idx
            
            if best_iou > self.iou_threshold:
                matches[tracker_id] = detections[best_detection_idx]
                used_detections.add(best_detection_idx)
        
        # Create new trackers for unmatched detections
        for idx, (bbox, keypoints) in enumerate(detections):
            if idx not in used_detections:
                # Initialize with 2.0s window duration
                new_tracker = PersonTracker(self.next_person_id, bbox, window_duration=2.0)
                self.trackers[self.next_person_id] = new_tracker
                matches[self.next_person_id] = (bbox, keypoints)
                self.next_person_id += 1
        
        return matches
    
    def classify_person(self, feature_window: np.ndarray) -> Tuple[str, float]:
        """
        Classify action from feature window.
        
        Args:
            feature_window: Array of shape (window_size, 17, 2)
            
        Returns:
            Tuple of (label, confidence)
        """
        # Prepare input for CNN-LSTM model
        # Expected shape: (batch, window_size, 17, 2, 1)
        input_data = feature_window[..., np.newaxis]  # Add channel dimension: (64, 17, 2, 1)
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension: (1, 64, 17, 2, 1)
        
        # Run inference
        predictions = self.cnn_model.predict(input_data, verbose=0)[0]
        
        # Get best prediction
        best_idx = np.argmax(predictions)
        confidence = predictions[best_idx]
        label = self.labels[best_idx]
        
        return label, confidence
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process single frame and return annotated frame.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Annotated frame with bounding boxes and labels
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Detect people
        detections = self.detect_people(frame)
        
        # Match to trackers
        matches = self.match_detections_to_trackers(detections, current_time)
        
        # Update trackers and classify
        for tracker_id, (bbox, keypoints) in matches.items():
            tracker = self.trackers[tracker_id]
            tracker.update(bbox, keypoints, current_time)
            
            # Try to classify if buffer is full
            feature_window = tracker.get_feature_window(target_frames=64)
            if feature_window is not None:
                label, confidence = self.classify_person(feature_window)
                if confidence > self.confidence_threshold:
                    tracker.current_label = label
                    tracker.confidence = confidence
        
        # Remove inactive trackers
        inactive_ids = [
            tid for tid, tracker in self.trackers.items()
            if not tracker.is_active(current_time)
        ]
        for tid in inactive_ids:
            del self.trackers[tid]
        
        # Draw annotations
        annotated_frame = self.draw_annotations(frame)
        
        # Calculate FPS
        fps = 1.0 / (current_time - self.last_frame_time + 1e-6)
        self.fps_history.append(fps)
        self.last_frame_time = current_time
        
        return annotated_frame
    
    def draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        """Draw bounding boxes, labels, and keypoints on frame."""
        annotated = frame.copy()
        
        for tracker in self.trackers.values():
            if not tracker.is_active(self.frame_count):
                continue
            
            bbox = tracker.bbox
            color = tracker.color
            
            # Draw bounding box
            cv2.rectangle(
                annotated,
                (int(bbox['xtl']), int(bbox['ytl'])),
                (int(bbox['xbr']), int(bbox['ybr'])),
                color,
                3
            )
            
            # Draw label with confidence
            label_text = f"ID{tracker.id}: {tracker.current_label}"
            if tracker.confidence > 0:
                label_text += f" ({tracker.confidence:.2f})"
            
            # Label background
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                annotated,
                (int(bbox['xtl']), int(bbox['ytl']) - label_size[1] - 10),
                (int(bbox['xtl']) + label_size[0] + 10, int(bbox['ytl'])),
                color,
                -1
            )
            
            # Label text
            cv2.putText(
                annotated,
                label_text,
                (int(bbox['xtl']) + 5, int(bbox['ytl']) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Draw keypoints (if recent)
            if len(tracker.keypoints_buffer) > 0:
                # Unpack timestamp and keypoints from the last buffer entry
                _, recent_keypoints = tracker.keypoints_buffer[-1]
                height, width = frame.shape[:2]
                
                for kp in recent_keypoints:
                    y, x, conf = kp
                    if conf > 0.3:
                        px = int(x * width)
                        py = int(y * height)
                        cv2.circle(annotated, (px, py), 4, color, -1)
                        cv2.circle(annotated, (px, py), 4, (255, 255, 255), 1)
        
        # Draw FPS and frame info
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        info_text = f"FPS: {avg_fps:.1f} | People: {len(self.trackers)} | Frame: {self.frame_count}"
        cv2.putText(
            annotated,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return annotated
    
    def process_video(
        self,
        video_source: str,
        output_path: Optional[str] = None,
        display: bool = True,
        max_frames: Optional[int] = None
    ):
        """
        Process video file or webcam stream.
        
        Args:
            video_source: Video file path or 'webcam' for webcam
            output_path: Optional path to save output video
            display: Whether to display video while processing
            max_frames: Maximum number of frames to process (None = all)
        """
        # Open video source
        if video_source.lower() == 'webcam':
            cap = cv2.VideoCapture(0)
            source_name = "Webcam"
        else:
            cap = cv2.VideoCapture(video_source)
            source_name = os.path.basename(video_source)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {video_source}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n{'='*60}")
        print(f"Processing: {source_name}")
        print(f"Resolution: {width}x{height} @ {fps:.2f} FPS")
        if total_frames > 0:
            print(f"Total frames: {total_frames}")
        print(f"{'='*60}\n")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Saving output to: {output_path}")
        
        # Reset state
        self.trackers = {}
        self.next_person_id = 0
        self.frame_count = 0
        
        # Process frames
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame = self.process_frame(frame)
                
                # Save frame
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame
                if display:
                    cv2.imshow('Real-time Pose Classification', annotated_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nStopped by user")
                        break
                    elif key == ord('p'):
                        print("Paused. Press any key to continue...")
                        cv2.waitKey(0)
                
                # Check max frames
                if max_frames and self.frame_count >= max_frames:
                    break
                
                # Progress update
                if self.frame_count % 100 == 0:
                    avg_fps = np.mean(self.fps_history) if self.fps_history else 0
                    print(f"Processed {self.frame_count} frames | FPS: {avg_fps:.1f}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        print(f"\n✓ Processing complete!")
        print(f"  Total frames processed: {self.frame_count}")
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        print(f"  Average FPS: {avg_fps:.1f}")


def load_labels(labels_path: str) -> List[str]:
    """Load class labels from file."""
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


def main():
    parser = argparse.ArgumentParser(
        description='Real-time pose classification with MoveNet + CNN'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained CNN model (.h5, .keras, or SavedModel)'
    )
    
    parser.add_argument(
        '--labels',
        type=str,
        required=True,
        help='Path to labels file (one label per line)'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='webcam',
        help='Video source: "webcam", video file path, or directory'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output video path (optional)'
    )
    
    parser.add_argument(
        '--window-size',
        type=int,
        default=64,
        help='Sliding window size (must match training, default: 64)'
    )
    
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.3,
        help='IoU threshold for person matching (default: 0.3)'
    )
    
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.5,
        help='Minimum confidence for classification (default: 0.5)'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable video display (useful for headless processing)'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        help='Maximum frames to process (optional)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.mp4',
        help='File pattern for directory processing (default: *.mp4)'
    )
    
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='Force CPU usage'
    )
    
    args = parser.parse_args()
    
    # Force CPU if requested
    if args.cpu_only:
        tf.config.set_visible_devices([], 'GPU')
        print("⚠ Running in CPU-only mode")
    
    # Load labels
    if not os.path.exists(args.labels):
        print(f"Error: Labels file not found: {args.labels}")
        return 1
    
    labels = load_labels(args.labels)
    print(f"Loaded {len(labels)} class labels: {labels}")
    
    # Load model and create classifier
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    classifier = RealtimeClassifier(
        model_path=args.model,
        labels=labels,
        window_size=args.window_size,
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence_threshold
    )
    
    # Process video(s)
    source_path = Path(args.source)
    
    if args.source.lower() == 'webcam':
        # Webcam mode
        classifier.process_video(
            'webcam',
            output_path=args.output,
            display=not args.no_display,
            max_frames=args.max_frames
        )
    
    elif source_path.is_file():
        # Single video file
        output_path = args.output
        if not output_path and not args.no_display:
            # Auto-generate output filename
            output_path = str(source_path.parent / f"{source_path.stem}_classified.mp4")
        
        classifier.process_video(
            str(source_path),
            output_path=output_path,
            display=not args.no_display,
            max_frames=args.max_frames
        )
    
    elif source_path.is_dir():
        # Directory of videos
        video_files = list(source_path.glob(args.pattern))
        
        if len(video_files) == 0:
            print(f"No videos found matching '{args.pattern}' in {args.source}")
            return 1
        
        print(f"\nFound {len(video_files)} video(s) to process")
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\n{'='*60}")
            print(f"Video {i}/{len(video_files)}: {video_file.name}")
            print(f"{'='*60}")
            
            output_path = None
            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(output_dir / f"{video_file.stem}_classified.mp4")
            
            try:
                classifier.process_video(
                    str(video_file),
                    output_path=output_path,
                    display=not args.no_display,
                    max_frames=args.max_frames
                )
            except Exception as e:
                print(f"Error processing {video_file.name}: {e}")
                continue
    
    else:
        print(f"Error: Invalid source: {args.source}")
        return 1
    
    print("\n✓ All done!")
    return 0


if __name__ == '__main__':
    exit(main())
