# MoveNet Keypoint Extraction for 2D CNN Training

This script extracts pose keypoints from video clips using MoveNet and converts them to 2D arrays suitable for CNN training.

## Features

- Parses CVAT XML annotation files to extract individual tracks (clips)
- **MultiPose detection**: Handles multiple people in the same frame
- **Intelligent person matching**: Uses IoU to match detections to annotated bounding boxes during training
- **Full frame processing**: Processes complete frames (no cropping) for better generalization
- Normalizes temporal dimension (2.0 seconds) to fixed frame count (64)
- Outputs 2D numpy arrays (51 features × 64 frames)
- Supports batch processing of multiple annotation files

## Usage
(**NOTE: A raw video and its annotaion should be put in the same folder**)
### 1. Process Single Annotation File

Extract keypoints from all tracks in a single CVAT annotation file:

```bash
python extract_keypoints_to_2d.py annotation --annotation data/train/raw/annotations_rear_hook_01.xml --output data/train/transformed/rear_hook/
```

### 2. Process All Annotation Files in Directory (recommended)

Process all annotation files matching a pattern:

```bash
python extract_keypoints_to_2d.py annotation-dir --input data/raw/rear_hook --output data/transformed/rear_hook/
```

### 3. Process Single Video Clip(without annotations)

Process entire video frame by frame (can only track one person):

```bash
python extract_keypoints_to_2d.py video --input data/raw/video.mp4 --output data/transformed/
```

## Command Line Arguments

### Annotation Mode

- `--annotation`: Path to CVAT XML annotation file (required)
- `--output`: Output directory for .npy files (required)
- `--frames`: Number of frames to normalize to (default: 64)
- `--model`: MoveNet model variant (default: movenet_multipose_lightning)
- `--cpu-only`: Force CPU usage even if GPU is available

### Annotation Directory Mode

- `--input`: Directory containing annotation XML files (required)
- `--output`: Output directory for .npy files (required)
- `--frames`: Number of frames to normalize to (default: 64)
- `--model`: MoveNet model variant (default: movenet_multipose_lightning)
- `--pattern`: File pattern for annotations (default: annotations_*.xml)
- `--cpu-only`: Force CPU usage even if GPU is available

### Video Mode

- `--input`: Video file or directory (required)
- `--output`: Output directory for .npy files (required)
- `--frames`: Number of frames to normalize to (default: 64)
- `--skip`: Process every Nth frame (default: 1)
- `--model`: MoveNet model variant (default: movenet_multipose_lightning)
- `--pattern`: File pattern for videos (default: *.mp4)
- `--cpu-only`: Force CPU usage even if GPU is available

## Output Files

For each track, the script generates:

1. **Keypoints file**: `{video_name}_{label}_{track_id}_keypoints.npy`
   - Shape: (51, num_frames)
   - Rows: 51 features (17 keypoints × 3 values: y, x, confidence)
   - Columns: Normalized frame count

2. **Metadata file**: `{video_name}_{label}_{track_id}_metadata.npy`
   - Contains track info, frame ranges, video properties, etc.

## Example Output Structure

```
transformed/
├── rear_hook_01_rear_hook_0_keypoints.npy
├── rear_hook_01_rear_hook_0_metadata.npy
├── rear_hook_01_rear_hook_1_keypoints.npy
├── rear_hook_01_rear_hook_1_metadata.npy
└── ...
```

## Settings Summary

- **Video Format**: MP4
- **Multiple People**: MultiPose model (detects up to 6 people per frame)
- **Temporal Representation**: Each frame as a column, normalized to fixed length
- **Keypoint Features**: 17 keypoints × 3 (y, x, confidence) = 51 features
- **Output Format**: NumPy .npy arrays
- **Model**: MoveNet MultiPose Lightning
- **Frame Processing**: Full frames (no cropping)
- **Person Matching**: IoU-based matching between detections and annotation bboxes
- **GPU Acceleration**: Automatic detection with memory growth enabled
- **Annotation Processing**:
  - Processes full frames with MultiPose
  - Uses annotation bboxes to identify correct person among multiple detections
  - Only uses frames with outside="0" (person visible)
  - Output naming: {video_name}_{label}_{track_id}_keypoints.npy

## MoveNet Keypoint Order

The 17 keypoints are:
0. nose, 1. left_eye, 2. right_eye, 3. left_ear, 4. right_ear,
5. left_shoulder, 6. right_shoulder, 7. left_elbow, 8. right_elbow,
9. left_wrist, 10. right_wrist, 11. left_hip, 12. right_hip,
13. left_knee, 14. right_knee, 15. left_ankle, 16. right_ankle

Each keypoint has 3 values: [y, x, confidence]
