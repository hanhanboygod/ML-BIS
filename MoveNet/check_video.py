"""
Quick test to verify annotations are being drawn correctly.
Saves a few annotated frames as images.
"""

import cv2
import numpy as np
import sys

print("Checking video file...")

video_path = "results/detected.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"ERROR: Cannot open {video_path}")
    print("The video file may be corrupted or encoded incorrectly.")
    sys.exit(1)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video info:")
print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps:.2f}")
print(f"  Total frames: {total_frames}")

if total_frames == 0:
    print("\n⚠ ERROR: Video has 0 frames!")
    print("The video writer failed to encode frames.")
    print("\nThis is a known issue with OpenCV's mp4v codec.")
    sys.exit(1)

# Read a few frames
test_frames = [100, 500, 1000, 2000, 3000]

print(f"\nReading test frames: {test_frames}")

for frame_num in test_frames:
    if frame_num >= total_frames:
        continue
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if ret:
        output_path = f"results/frame_{frame_num}.jpg"
        cv2.imwrite(output_path, frame)
        print(f"  ✓ Saved frame {frame_num} to {output_path}")
    else:
        print(f"  ✗ Failed to read frame {frame_num}")

cap.release()

print("\n✓ Check complete!")
print("\nIf frames are blank or show no annotations:")
print("  1. The video codec may not be compatible")
print("  2. Try re-running with --no-display to force frame writing")
print("  3. Use a different codec (avc1, XVID, or FFV1)")
