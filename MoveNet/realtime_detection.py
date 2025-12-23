"""
Real-time Person Detection and Tracking (No Classification)

This script performs real-time person detection and tracking using MoveNet MultiPose.
No CNN model required - just detects people and draws bounding boxes with IDs.

Features:
- Detects multiple people simultaneously
- Tracks people across frames with unique IDs
- Displays bounding boxes and keypoints
- Shows FPS and detection count

Usage:
    # Webcam
    python realtime_detection.py --source webcam
    
    # Video file
    python realtime_detection.py --source video.mp4 --output result.mp4
    
    # Video directory
    python realtime_detection.py --source videos/ --output results/ --pattern "*.mp4"

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
    """Track individual people across frames."""
    
    def __init__(self, person_id: int, initial_bbox: Dict):
        """
        Initialize person tracker.
        
        Args:
            person_id: Unique ID for this person
            initial_bbox: Initial bounding box {'xtl', 'ytl', 'xbr', 'ybr'}
        """
        self.id = person_id
        self.bbox = initial_bbox
        self.last_seen_frame = 0
        self.color = self._generate_color()
        self.keypoints_history = deque(maxlen=5)  # Store last 5 frames for display
    
    def _generate_color(self):
        """Generate a unique color for this person."""
        np.random.seed(self.id * 42)
        return tuple(map(int, np.random.randint(50, 255, 3)))
    
    def update(self, bbox: Dict, keypoints: np.ndarray, frame_num: int):
        """Update tracker with new detection."""
        self.bbox = bbox
        self.keypoints_history.append(keypoints)
        self.last_seen_frame = frame_num
    
    def is_active(self, current_frame: int, max_missing_frames: int = 30) -> bool:
        """Check if tracker is still active (recently seen)."""
        return (current_frame - self.last_seen_frame) <= max_missing_frames
    
    def get_recent_keypoints(self) -> Optional[np.ndarray]:
        """Get most recent keypoints."""
        return self.keypoints_history[-1] if self.keypoints_history else None


class RealtimeDetector:
    """Real-time person detection and tracking with MoveNet."""
    
    # MoveNet keypoint connections for drawing skeleton
    KEYPOINT_EDGES = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (0, 5), (0, 6),  # Shoulders
        (5, 7), (7, 9),  # Left arm
        (6, 8), (8, 10),  # Right arm
        (5, 6),  # Shoulders
        (5, 11), (6, 12),  # Torso
        (11, 12),  # Hips
        (11, 13), (13, 15),  # Left leg
        (12, 14), (14, 16),  # Right leg
    ]
    
    def __init__(
        self,
        iou_threshold: float = 0.3,
        detection_threshold: float = 0.5,
        min_keypoints: int = 6,
        verbose: bool = False
    ):
        """
        Initialize real-time detector.
        
        Args:
            iou_threshold: IoU threshold for person matching
            detection_threshold: Minimum detection score (default: 0.5)
            min_keypoints: Minimum number of confident keypoints (default: 6)
            verbose: Print debug information
        """
        self.iou_threshold = iou_threshold
        self.detection_threshold = detection_threshold
        self.min_keypoints = min_keypoints
        self.verbose = verbose
        
        # Load MoveNet
        print("Loading MoveNet MultiPose Lightning...")
        model_url = 'https://tfhub.dev/google/movenet/multipose/lightning/1'
        
        # Force model to use GPU if available
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            self.movenet_model = hub.load(model_url)
            self.movenet = self.movenet_model.signatures['serving_default']
        
        self.input_size = 256
        device_name = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
        print(f"✓ MoveNet loaded successfully on {device_name}!")
        
        # Test inference to warm up GPU
        print("  Warming up GPU with test inference...")
        test_img = tf.zeros((1, 256, 256, 3), dtype=tf.int32)
        _ = self.movenet(test_img)
        print("  ✓ GPU warmed up!")
        
        # Person tracking
        self.trackers: Dict[int, PersonTracker] = {}
        self.next_person_id = 0
        self.frame_count = 0
        
        # Performance metrics
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
    
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
        
        # Debug: Print detection scores if verbose
        if self.verbose and self.frame_count % 30 == 0:
            scores = [d[55] for d in keypoints_with_scores]
            print(f"  Frame {self.frame_count}: Detection scores: {[f'{s:.3f}' for s in scores[:3]]}")
        
        # Extract people with good confidence
        detections = []
        for i, detection in enumerate(keypoints_with_scores):
            detection_score = detection[55]  # Overall detection score
            
            if detection_score > self.detection_threshold:
                keypoints = detection[:51].reshape(17, 3)
                
                # Count confident keypoints (quality check)
                confident_kp_count = np.sum(keypoints[:, 2] > 0.3)
                
                # Filter out detections with too few confident keypoints
                if confident_kp_count < self.min_keypoints:
                    continue
                
                bbox = self.keypoints_to_bbox(keypoints, width, height)
                
                # Check if bbox is valid and reasonable size
                bbox_width = bbox['xbr'] - bbox['xtl']
                bbox_height = bbox['ybr'] - bbox['ytl']
                
                if bbox_width > 0 and bbox_height > 0:
                    # Filter out tiny detections (likely noise)
                    if bbox_width < 50 or bbox_height < 50:
                        continue
                    
                    detections.append((bbox, keypoints))
                    if self.verbose and len(detections) <= 3:
                        print(f"    Detection {i}: score={detection_score:.3f}, kp={confident_kp_count}/17, bbox={int(bbox_width)}x{int(bbox_height)}")
        
        return detections
    
    def match_detections_to_trackers(
        self,
        detections: List[Tuple[Dict, np.ndarray]]
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
            if not tracker.is_active(self.frame_count):
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
                new_tracker = PersonTracker(self.next_person_id, bbox)
                self.trackers[self.next_person_id] = new_tracker
                matches[self.next_person_id] = (bbox, keypoints)
                self.next_person_id += 1
        
        return matches
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process single frame and return annotated frame.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Annotated frame with bounding boxes and keypoints
        """
        self.frame_count += 1
        
        # Detect people
        detections = self.detect_people(frame)
        
        # Match to trackers
        matches = self.match_detections_to_trackers(detections)
        
        # Update trackers
        for tracker_id, (bbox, keypoints) in matches.items():
            tracker = self.trackers[tracker_id]
            tracker.update(bbox, keypoints, self.frame_count)
        
        # Remove inactive trackers
        inactive_ids = [
            tid for tid, tracker in self.trackers.items()
            if not tracker.is_active(self.frame_count)
        ]
        for tid in inactive_ids:
            del self.trackers[tid]
        
        # Draw annotations
        annotated_frame = self.draw_annotations(frame)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time + 1e-6)
        self.fps_history.append(fps)
        self.last_frame_time = current_time
        
        return annotated_frame
    
    def draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        """Draw bounding boxes, labels, and keypoints on frame."""
        annotated = frame.copy()
        height, width = frame.shape[:2]
        
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
            
            # Draw label
            label_text = f"Person ID: {tracker.id}"
            
            # Label background
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(
                annotated,
                (int(bbox['xtl']), int(bbox['ytl']) - label_size[1] - 15),
                (int(bbox['xtl']) + label_size[0] + 10, int(bbox['ytl'])),
                color,
                -1
            )
            
            # Label text
            cv2.putText(
                annotated,
                label_text,
                (int(bbox['xtl']) + 5, int(bbox['ytl']) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Draw skeleton
            keypoints = tracker.get_recent_keypoints()
            if keypoints is not None:
                self.draw_skeleton(annotated, keypoints, width, height, color)
        
        # Draw FPS and frame info
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        info_text = f"FPS: {avg_fps:.1f} | People: {len(self.trackers)} | Frame: {self.frame_count}"
        
        # Info background
        info_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(
            annotated,
            (5, 5),
            (info_size[0] + 15, info_size[1] + 20),
            (0, 0, 0),
            -1
        )
        
        # Info text
        cv2.putText(
            annotated,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        
        return annotated
    
    def draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        width: int,
        height: int,
        color: Tuple[int, int, int]
    ):
        """Draw pose skeleton on frame."""
        # Draw keypoint connections
        for edge in self.KEYPOINT_EDGES:
            kp1 = keypoints[edge[0]]
            kp2 = keypoints[edge[1]]
            
            if kp1[2] > 0.3 and kp2[2] > 0.3:  # Both keypoints confident
                y1, x1 = int(kp1[0] * height), int(kp1[1] * width)
                y2, x2 = int(kp2[0] * height), int(kp2[1] * width)
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw keypoints
        for kp in keypoints:
            y, x, conf = kp
            if conf > 0.3:
                px = int(x * width)
                py = int(y * height)
                cv2.circle(frame, (px, py), 5, color, -1)
                cv2.circle(frame, (px, py), 5, (255, 255, 255), 1)
    
    def process_video(
        self,
        video_source: str,
        output_path: Optional[str] = None,
        display: bool = True,
        max_frames: Optional[int] = None,
        save_frames: bool = False
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
            # Try different codecs for better compatibility
            # 'avc1' for H.264 encoding (better compatibility)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not writer.isOpened():
                print("⚠ H.264 codec failed, trying XVID...")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if writer.isOpened():
                print(f"Saving output to: {output_path}")
            else:
                print(f"⚠ WARNING: VideoWriter failed to open! Output may not be saved correctly.")
                writer = None
        
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
                
                # Save frame to video
                if writer:
                    writer.write(annotated_frame)
                
                # Save sample frames as images
                if save_frames and self.frame_count % 500 == 0:
                    if output_path:
                        frame_dir = Path(output_path).parent
                        frame_path = frame_dir / f"frame_{self.frame_count:05d}.jpg"
                        cv2.imwrite(str(frame_path), annotated_frame)
                        print(f"  Saved frame {self.frame_count} to {frame_path}")
                
                # Display frame
                if display:
                    cv2.imshow('Real-time Person Detection', annotated_frame)
                    
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


def main():
    parser = argparse.ArgumentParser(
        description='Real-time person detection and tracking with MoveNet (no classification)'
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
        '--iou-threshold',
        type=float,
        default=0.3,
        help='IoU threshold for person matching (default: 0.3)'
    )
    
    parser.add_argument(
        '--detection-threshold',
        type=float,
        default=0.5,
        help='Minimum detection score (default: 0.5, range: 0.1-0.8)'
    )
    
    parser.add_argument(
        '--min-keypoints',
        type=int,
        default=6,
        help='Minimum number of confident keypoints (default: 6)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print debug information'
    )
    
    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='Save sample annotated frames as JPG images (every 500 frames)'
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
    
    # Create detector
    detector = RealtimeDetector(
        iou_threshold=args.iou_threshold,
        detection_threshold=args.detection_threshold,
        min_keypoints=args.min_keypoints,
        verbose=args.verbose
    )
    
    # Process video(s)
    source_path = Path(args.source)
    
    if args.source.lower() == 'webcam':
        # Webcam mode
        detector.process_video(
            'webcam',
            output_path=args.output,
            display=not args.no_display,
            max_frames=args.max_frames,
            save_frames=args.save_frames
        )
    
    elif source_path.is_file():
        # Single video file
        output_path = args.output
        if not output_path and not args.no_display:
            # Auto-generate output filename
            output_path = str(source_path.parent / f"{source_path.stem}_detected.mp4")
        
        detector.process_video(
            str(source_path),
            output_path=output_path,
            display=not args.no_display,
            max_frames=args.max_frames,
            save_frames=args.save_frames
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
                output_path = str(output_dir / f"{video_file.stem}_detected.mp4")
            
            try:
                detector.process_video(
                    str(video_file),
                    output_path=output_path,
                    display=not args.no_display,
                    max_frames=args.max_frames,
                    save_frames=args.save_frames
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
