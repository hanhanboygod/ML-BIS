# Real-time Detection & Classification Guide

## Overview

This guide covers two scripts for real-time pose analysis:

1. **`realtime_detection.py`** - Person detection and tracking (no model required)
2. **`realtime_inference.py`** - Full classification with trained CNN model

Both use **MoveNet MultiPose** for detecting and tracking multiple people simultaneously.

## How It Works

```
┌─────────────┐
│ Video Frame │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│  MoveNet MultiPose Detection    │
│  (Detects up to 6 people)       │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Person Tracking (IoU matching) │
│  - Match detections to trackers │
│  - Create new trackers          │
│  - Remove inactive trackers     │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Sliding Window Buffer          │
│  - Maintain 2.0s history/person │
│  - Update with new keypoints    │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Feature Extraction             │
│  - Resample to (51, 64) array   │
│  - Normalize like training data │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  CNN Classification             │
│  - Run inference on features    │
│  - Get action label + confidence│
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Visualization                  │
│  - Draw bounding boxes          │
│  - Display labels & confidence  │
│  - Show keypoints               │
└─────────────────────────────────┘
```

## Key Features

### 1. Multi-Person Detection
- Detects up to 6 people per frame using MoveNet MultiPose
- Each person tracked independently with unique ID and color

### 2. Person Tracking
- **IoU-based matching**: Tracks same person across frames
- **Automatic ID assignment**: New people get new IDs
- **Inactive tracker removal**: Removes people who disappear for >1.0 second

### 3. Sliding Window Classification
- Maintains **2.0-second buffer** per person (time-based)
- **Dynamic Resampling**: Interpolates variable frame counts to fixed 64 frames
- Classifies action when buffer fills
- Updates in real-time as new frames arrive

### 4. Visual Feedback
- **Bounding boxes**: Colored boxes around each person
- **Labels**: Action name + confidence score
- **Keypoints**: Pose skeleton overlay
- **FPS counter**: Real-time performance metric
- **Person count**: Number of tracked people

---

## Part 1: Real-time Detection (No Model Required)

### Quick Start with `realtime_detection.py`

Test person detection and tracking **before** training your CNN model:

```bash
# In WSL2
cd /mnt/d/.../MoveNet
source movenet_env_wsl/bin/activate

# Process video file
python realtime_detection.py \
  --source path/to/video \
  --output path/to/results/*.mp4
```

### Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--detection-threshold` | 0.5 | Minimum detection confidence (0.1-0.8) |
| `--min-keypoints` | 6 | Minimum confident keypoints required |
| `--iou-threshold` | 0.3 | IoU threshold for person tracking |
| `--verbose` | False | Print detection details |
| `--save-frames` | False | Save sample frames as JPG |

### Example: Stricter Detection

```bash
# Reduce false positives
python realtime_detection.py \
  --source video.mp4 \
  --detection-threshold 0.6 \
  --min-keypoints 8
```

### Example: More Sensitive Detection

```bash
# Detect more people (may include false positives)
python realtime_detection.py \
  --source video.mp4 \
  --detection-threshold 0.3 \
  --min-keypoints 4
```

### Troubleshooting Detection

**Too many false positives?**
- Increase `--detection-threshold` (0.5 → 0.6)
- Increase `--min-keypoints` (6 → 8)

**Missing real people?**
- Decrease `--detection-threshold` (0.5 → 0.3)
- Decrease `--min-keypoints` (6 → 4)

**Check what's being detected:**
```bash
# Use verbose mode
python realtime_detection.py \
  --source video.mp4 \
  --verbose
```

---

## Part 2: Real-time Classification (With Trained Model)

### Installation

```bash
# Activate your environment
source movenet_env_wsl/bin/activate  # WSL2/Linux

# Dependencies should already be installed:
# - tensorflow (with CUDA for GPU)
# - tensorflow-hub
# - opencv-python
# - numpy
```

## Classification Usage Examples

### 1. Prepare Your Model & Labels

```bash
# Create labels.txt with your action classes
echo "cross
uppercut
rear_hook
jab" > labels.txt
```

**Important:** Label order must match your training class order!

### 2. Single Video File

```bash
python realtime_inference.py \
  --model trained_models/boxing_classifier.h5 \
  --labels labels.txt \
  --source videos/sparring_match.mp4 \
  --output results/sparring_classified.mp4
```

### 3. Video Directory (Batch Processing)

```bash
python realtime_inference.py \
  --model trained_models/boxing_classifier.h5 \
  --labels labels.txt \
  --source videos/ \
  --output results/ \
  --pattern "*.mp4" \
  --no-display
```

### 4. Custom Parameters

```bash
python realtime_inference.py \
  --model trained_models/boxing_classifier.h5 \
  --labels labels.txt \
  --source sparring.mp4 \
  --window-size 64 \
  --iou-threshold 0.3 \
  --confidence-threshold 0.7 \
  --max-frames 1000
```

## Command-Line Arguments

### Detection Script (`realtime_detection.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source` | str | **required** | Video source: file path, or directory |
| `--output` | str | None | Output video path |
| `--detection-threshold` | float | 0.5 | Minimum detection confidence (0.1-0.8) |
| `--min-keypoints` | int | 6 | Minimum confident keypoints (3-15) |
| `--iou-threshold` | float | 0.3 | IoU threshold for tracking |
| `--verbose` | flag | False | Print detection details |
| `--save-frames` | flag | False | Save sample frames as JPG |
| `--no-display` | flag | False | Disable video window |
| `--max-frames` | int | None | Maximum frames to process |
| `--pattern` | str | `*.mp4` | File pattern for directories |
| `--cpu-only` | flag | False | Force CPU usage |

### Classification Script (`realtime_inference.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | **required** | Path to trained CNN model (.h5, .keras, SavedModel) |
| `--labels` | str | **required** | Path to labels file (one label per line) |
| `--source` | str | **required** | Video source: file path, or directory |
| `--output` | str | None | Output video path (auto-generated if not specified) |
| `--window-size` | int | 64 | Sliding window size (must match training) |
| `--iou-threshold` | float | 0.3 | IoU threshold for person matching |
| `--confidence-threshold` | float | 0.5 | Minimum confidence to display prediction |
| `--no-display` | flag | False | Disable video display (headless mode) |
| `--max-frames` | int | None | Maximum frames to process |
| `--pattern` | str | `*.mp4` | File pattern for directory processing |
| `--cpu-only` | flag | False | Force CPU usage (disable GPU) |

## Labels File Format

Create a text file with one class label per line:

```
# labels.txt
cross
uppercut
rear_hook
jab

```

**Order matters!** Labels must match the order used during training.

## Performance Optimization

## Tracking Algorithm

### Person Matching Strategy

Each frame:
1. **Detect** all people using MoveNet
2. **Calculate IoU** between detections and existing trackers
3. **Match** detections to trackers (highest IoU > threshold)
4. **Create** new trackers for unmatched detections
5. **Remove** trackers not seen for 30+ frames

### Why IoU Matching?

- **Robust to movement**: Works even when people move quickly
- **Handles occlusion**: Temporarily lost people can be re-matched
- **Simple and fast**: No complex deep learning required

## Troubleshooting

**Solutions:**
- Increase `--iou-threshold` (try 0.5)
- Reduce occlusion in video
- Improve lighting/video quality

### Issue: Wrong Classifications

**Solutions:**
- Check labels order matches training
- Increase `--confidence-threshold` to filter weak predictions
- Retrain model with more data
- Verify `--window-size` matches training

### Issue: People Not Detected

**Solutions:**
- Check video resolution (MoveNet works best at 720p+)
- Ensure good lighting
- People should be visible (not too small in frame)
- Lower detection threshold in code (currently 0.1)

### Issue: Codec not found
When trying to execute realtime_inference.py:
```
[ERROR:0@44.106] global cap_ffmpeg_impl.hpp:3207 open Could not find encoder for codec_id=27, error: Encoder not found
[ERROR:0@44.106] global cap_ffmpeg_impl.hpp:3285 open VIDEOIO/FFMPEG: Failed to initialize VideoWriter
⚠ H.264 codec failed, trying XVID...
OpenCV: FFMPEG: tag 0x44495658/'XVID' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
```
**Solutions:**
```
# Install FFmpeg with x264 encoder
sudo apt-get update
sudo apt-get install -y ffmpeg libx264-dev

# Verify H.264 codec is available
ffmpeg -codecs | grep 264
```

## Output Video Format

Output videos include:
- **Bounding boxes**: Colored boxes around each person
- **Labels**: "ID{N}: {action} ({confidence})"
- **Keypoints**: Red dots showing pose skeleton
- **FPS counter**: Top-left corner
- **People count**: Number of tracked people
- **Frame number**: Current frame

Example output label: `ID0: jab (0.87)`

## Integration with Training Pipeline

### Complete Workflow

```bash
# 1. Extract keypoints from annotated videos
python extract_keypoints_to_2d.py annotation-dir \
  --input data/train/raw/ \
  --output data/train/transformed/ \
  --frames 64

# 2. Train CNN model (your training script)

# 3. Run real-time inference
python realtime_inference.py \
  --model trained_models/boxing_classifier.h5 \
  --labels labels.txt \
  --source test_video.mp4 \
  --output results/test_classified.mp4
```

## Advanced: Modifying the Script

### Adjust Detection Sensitivity

Edit line ~230 in `realtime_inference.py`:
```python
if detection_score > 0.1:  # Change this threshold
```

## Example Output

```
============================================================
Processing: sparring_match.mp4
Resolution: 1280x720 @ 30.00 FPS
Total frames: 900
============================================================

Processed 100 frames | FPS: 45.2
Processed 200 frames | FPS: 46.8
Processed 300 frames | FPS: 45.5
...

✓ Processing complete!
  Total frames processed: 900
  Average FPS: 45.8
```

---

