# MoveNet Keypoint Extraction for 2D CNN Training

This script extracts pose keypoints from video clips using MoveNet and converts them to 2D arrays suitable for CNN training.

## Features

- Parses CVAT XML annotation files to extract individual tracks (clips)
- **MultiPose detection**: Handles multiple people in the same frame
- **Intelligent person matching**: Uses IoU to match detections to annotated bounding boxes during training
- **Full frame processing**: Processes complete frames (no cropping) for better generalization
- Normalizes temporal dimension to fixed frame count
- Outputs 2D numpy arrays (51 features × N frames)
- Supports batch processing of multiple annotation files

## Training vs Inference Workflow

### Training Phase (with annotations):
1. Process full frames with MultiPose (detects all people)
2. Use annotation bounding boxes to **match and identify** the correct person
3. Extract keypoints for the annotated person only
4. Generate labeled training data

### Inference Phase (without annotations):
1. Process frames with MultiPose (detects all people)
2. Extract keypoints for **all detected people**
3. Feed each person's keypoints to trained CNN
4. Get classification for each person independently

This approach ensures your model can classify multiple people performing different actions in the same frame!

## Installation & Environment Setup

### ⚠️ IMPORTANT: GPU Support on Windows

**TensorFlow on Windows does NOT support GPU natively.** The official Windows binaries are CPU-only (2-5 FPS).

For GPU acceleration (30-100 FPS, 10-20x faster), **use WSL2** on Windows.

---

## Complete Setup Guide

### Option 1: Windows with WSL2 (Recommended for GPU) ✅

**System Requirements:**
- Windows 10/11 with WSL2 installed
- NVIDIA GPU (e.g., RTX 4060)
- NVIDIA driver installed in Windows

#### Step 1: Verify WSL2 and GPU
```powershell
# In Windows PowerShell - Check WSL2 is installed
wsl --list --verbose

# Check GPU is accessible in WSL2
wsl -d Ubuntu nvidia-smi
```

You should see your GPU listed. If not, install WSL2:
```powershell
wsl --install Ubuntu
```

#### Step 2: Navigate to Project in WSL2
```bash
# Open WSL2 Ubuntu terminal (or use: wsl -d Ubuntu)
# Your Windows files are at /mnt/DRIVE/...
cd /mnt/ ... /MoveNet_test
```

#### Step 3: Install System Dependencies
```bash
# Update system
sudo apt-get update

# Install Python 3.11
sudo apt-get install -y python3.11 python3.11-venv python3-pip

# Install OpenCV dependencies (required for cv2)
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
```

#### Step 4: Create Virtual Environment
```bash
# Create environment (in your project directory)
python3.11 -m venv movenet_env_wsl

# Activate it
source movenet_env_wsl/bin/activate
```

You should see `(movenet_env_wsl)` in your prompt.

#### Step 5: Install Python Packages
```bash
# Upgrade pip
pip install --upgrade pip

# Install TensorFlow with CUDA support + other dependencies
pip install tensorflow[and-cuda] tensorflow-hub opencv-python numpy
```

This installs:
- TensorFlow 2.20+ with full GPU support
- CUDA libraries (automatically)
- cuDNN (automatically)
- TensorFlow Hub for MoveNet models
- OpenCV for video processing
- NumPy for array operations

#### Step 6: Verify GPU Detection
```bash
# Check GPU is detected by TensorFlow
python check_gpu.py
```

**Expected output:**
```
✓ GPU acceleration enabled: 1 Physical GPU(s), 1 Logical GPU(s)
  GPU 0: /physical_device:GPU:0
------------------------------------------------------------
Loading movenet_multipose_lightning model on GPU...
✓ Model loaded successfully on GPU! Input size: 256x256
  MultiPose mode: True
```

✅ **Setup Complete!** You're ready to process videos with GPU acceleration.

#### Daily Usage (WSL2)
```bash
# 1. Open WSL2 terminal
wsl -d Ubuntu

# 2. Navigate to project
cd /mnt/ ... /MoveNet_test

# 3. Activate environment
source movenet_env_wsl/bin/activate

# 4. Run your scripts
python extract_keypoints_to_2d.py annotation-dir \
    --input data/train/raw/ \
    --output data/train/transformed/
```

**Pro Tips:**
- Edit files in Windows (VS Code, File Explorer work normally)
- Run scripts in WSL2 with GPU
- All changes sync instantly (same files, no copying)
- Access WSL2 files from Windows: `\\wsl$\Ubuntu\home\USERNAME\`

---

### Option 2: Windows Native (CPU Only)


```powershell
# Create virtual environment (in PowerShell)
python -m venv movenet_env
movenet_env\Scripts\activate

# Install packages (CPU-only version)
pip install tensorflow tensorflow-hub opencv-python numpy
```

---

### Option 3: Linux Native (GPU Support)

```bash
# Create virtual environment
python3 -m venv movenet_env
source movenet_env/bin/activate

# Install with GPU support
pip install tensorflow[and-cuda] tensorflow-hub opencv-python numpy

# Verify GPU
python check_gpu.py
```

---

### Option 4: macOS (CPU Only)

```bash
# Create virtual environment
python3 -m venv movenet_env
source movenet_env/bin/activate

# Install packages
pip install tensorflow tensorflow-hub opencv-python numpy
```

## Troubleshooting

### WSL2: GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi  # Should show your GPU

# Reinstall TensorFlow with CUDA
pip uninstall tensorflow
pip install tensorflow[and-cuda]
```

### WSL2: OpenCV Import Error
```bash
# Install system libraries
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### WSL2: Environment Not Found
```bash
# Make sure you're in the right directory
cd /mnt/d/NTHUcourses/Junior_autumn/Introduction_to_Machine_Learning/Final/MoveNet_test

# Activate environment
source movenet_env_wsl/bin/activate
```

### Windows: "python" not found
```bash
# In WSL2, use python3 or activate environment first
python3 --version

# Or activate environment
source movenet_env_wsl/bin/activate
python --version  # Now works
```

### Permission Denied
```bash
# Fix permissions
chmod +x wsl2_setup_direct.sh
chmod -R 755 .
```

## Real-Time Inference

After extracting keypoints and training your CNN model, you can perform real-time pose classification on videos or webcam streams:

### Detection Only (No Model Required)
Test detection and tracking before training your model:
```bash
python realtime_detection.py --source data/train/raw/rear_hook_01.mp4 --output results/detected.mp4
```

### Full Classification (Requires Trained Model)
Perform real-time action classification:
```bash
python realtime_inference.py --model trained_model.h5 --labels labels_example.txt --source video.mp4
```

📖 **See [REALTIME_INFERENCE_GUIDE.md](REALTIME_INFERENCE_GUIDE.md) for complete documentation.**

---

## Usage

### 1. Process Single Annotation File

Extract keypoints from all tracks in a single CVAT annotation file:

```bash
python extract_keypoints_to_2d.py annotation --annotation data/train/raw/annotations_rear_hook_01.xml --output data/train/transformed
```

### 2. Process All Annotation Files in Directory

Process all annotation files matching a pattern:

```bash
python extract_keypoints_to_2d.py annotation-dir --input data/train/raw/ --output data/train/transformed/
```

### 3. Process Raw Video (without annotations)

Process entire video frame by frame (can only track one person):

```bash
python extract_keypoints_to_2d.py video --input data/valid/raw/video.mp4 --output data/valid/transformed
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
output/
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

## Loading Output for Training

```python
import numpy as np

# Load keypoints
keypoints = np.load('rear_hook_01_rear_hook_0_keypoints.npy')
print(f"Shape: {keypoints.shape}")  # (51, 64)

# Load metadata
metadata = np.load('rear_hook_01_rear_hook_0_metadata.npy', allow_pickle=True).item()
print(f"Label: {metadata['label']}")
print(f"Track ID: {metadata['track_id']}")
print(f"Frame range: {metadata['frame_range']}")
```

## MoveNet Keypoint Order

The 17 keypoints are:
0. nose, 1. left_eye, 2. right_eye, 3. left_ear, 4. right_ear,
5. left_shoulder, 6. right_shoulder, 7. left_elbow, 8. right_elbow,
9. left_wrist, 10. right_wrist, 11. left_hip, 12. right_hip,
13. left_knee, 14. right_knee, 15. left_ankle, 16. right_ankle

Each keypoint has 3 values: [y, x, confidence]
