"""
Debug script to test MoveNet detection on a single frame from video.
This helps diagnose detection issues.
"""

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


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


print("=" * 60)
print("MoveNet Detection Debug Tool")
print("=" * 60)

GPU_AVAILABLE = configure_gpu()
print("-" * 60)

# Video to test
VIDEO_PATH = "data/train/raw/rear_hook_01.mp4"
FRAME_TO_TEST = 100  # Test frame 100

print(f"\nOpening video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERROR: Cannot open video {VIDEO_PATH}")
    exit(1)

# Get video info
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video info:")
print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps:.2f}")
print(f"  Total frames: {total_frames}")

# Read specific frame
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_TO_TEST)
ret, frame = cap.read()
cap.release()

if not ret:
    print(f"ERROR: Cannot read frame {FRAME_TO_TEST}")
    exit(1)

print(f"\n✓ Successfully read frame {FRAME_TO_TEST}")
print(f"  Frame shape: {frame.shape}")

# Load MoveNet
print("\nLoading MoveNet MultiPose Lightning...")
model_url = 'https://tfhub.dev/google/movenet/multipose/lightning/1'
movenet_model = hub.load(model_url)
movenet = movenet_model.signatures['serving_default']
print("✓ MoveNet loaded successfully!")

# Prepare frame
print("\nProcessing frame...")
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img = tf.image.resize_with_pad(
    tf.expand_dims(frame_rgb, axis=0),
    256,
    256
)
input_image = tf.cast(img, dtype=tf.int32)

print(f"  Input shape: {input_image.shape}")
print(f"  Input dtype: {input_image.dtype}")

# Run detection
print("\nRunning MoveNet detection...")
outputs = movenet(input_image)
keypoints_with_scores = outputs['output_0'].numpy()[0]

print(f"✓ Detection complete!")
print(f"  Output shape: {keypoints_with_scores.shape}")

# Analyze detections
print("\n" + "=" * 60)
print("DETECTION RESULTS")
print("=" * 60)

for i, detection in enumerate(keypoints_with_scores):
    detection_score = detection[55]
    
    # Extract keypoints
    keypoints = detection[:51].reshape(17, 3)
    
    # Count confident keypoints
    confident_kp = np.sum(keypoints[:, 2] > 0.3)
    avg_confidence = np.mean(keypoints[:, 2])
    
    # Calculate bbox
    confident_kp_coords = keypoints[keypoints[:, 2] > 0.3]
    if len(confident_kp_coords) > 0:
        y_coords = confident_kp_coords[:, 0] * height
        x_coords = confident_kp_coords[:, 1] * width
        bbox_width = np.max(x_coords) - np.min(x_coords)
        bbox_height = np.max(y_coords) - np.min(y_coords)
        bbox_area = bbox_width * bbox_height
    else:
        bbox_width = bbox_height = bbox_area = 0
    
    print(f"\nDetection {i}:")
    print(f"  Detection score: {detection_score:.4f}")
    print(f"  Confident keypoints: {confident_kp}/17")
    print(f"  Avg keypoint confidence: {avg_confidence:.4f}")
    print(f"  Bbox size: {bbox_width:.0f}x{bbox_height:.0f} (area: {bbox_area:.0f})")
    
    if detection_score > 0.05:
        print(f"  → WOULD BE DETECTED (threshold=0.05) ✓")
    else:
        print(f"  → Would be filtered out (threshold=0.05)")

# Find best detection
best_idx = np.argmax(keypoints_with_scores[:, 55])
best_score = keypoints_with_scores[best_idx, 55]

print("\n" + "=" * 60)
print(f"BEST DETECTION: #{best_idx} with score {best_score:.4f}")
print("=" * 60)

# Visualize best detection
best_keypoints = keypoints_with_scores[best_idx, :51].reshape(17, 3)
vis_frame = frame.copy()

# Draw keypoints
for kp in best_keypoints:
    y, x, conf = kp
    if conf > 0.3:
        px = int(x * width)
        py = int(y * height)
        cv2.circle(vis_frame, (px, py), 5, (0, 255, 0), -1)
        cv2.circle(vis_frame, (px, py), 5, (255, 255, 255), 1)

# Draw bbox
confident_kp = best_keypoints[best_keypoints[:, 2] > 0.3]
if len(confident_kp) > 0:
    y_coords = confident_kp[:, 0] * height
    x_coords = confident_kp[:, 1] * width
    
    xtl = int(np.min(x_coords) - 20)
    ytl = int(np.min(y_coords) - 20)
    xbr = int(np.max(x_coords) + 20)
    ybr = int(np.max(y_coords) + 20)
    
    cv2.rectangle(vis_frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 3)
    cv2.putText(vis_frame, f"Score: {best_score:.3f}", (xtl, ytl-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Save visualization
output_path = "debug_detection.jpg"
cv2.imwrite(output_path, vis_frame)
print(f"\n✓ Visualization saved to: {output_path}")

# Show recommendations
print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

if best_score < 0.05:
    print("⚠ Very low detection score!")
    print("  Possible issues:")
    print("  - Person not visible in frame")
    print("  - Video quality too low")
    print("  - Person too small in frame")
    print("  - Heavy occlusion")
elif best_score < 0.15:
    print("⚠ Low detection score")
    print("  - Use --detection-threshold 0.05 (or lower)")
    print("  - Check video quality")
else:
    print("✓ Good detection score!")
    print("  - Default threshold (0.05) should work fine")

print("\nTo run realtime detection with adjusted threshold:")
print(f"  python realtime_detection.py \\")
print(f"    --source {VIDEO_PATH} \\")
print(f"    --detection-threshold 0.05 \\")
print(f"    --verbose")

print("\n✓ Debug complete!")
