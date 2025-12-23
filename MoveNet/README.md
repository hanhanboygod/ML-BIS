# MoveNet Keypoint Extraction & Realtime Inference

## Training vs Inference Workflow

### Training Phase (with annotations):
1. Process full frames with MultiPose (detects all people)
2. Use annotation bounding boxes to **match and identify** the correct person
3. Extract keypoints for the annotated person only
4. Generate labeled training data

### Inference Phase (without annotations):
1. Process frames with MultiPose (detects all people)
2. Extract keypoints for **all detected people**
3. Feed each person's keypoints to trained model
4. Get classification for each person independently


## Installation & Environment Setup

### ⚠️ IMPORTANT: GPU Support on Windows

**TensorFlow on Windows does NOT support GPU natively.** 

---

## Complete Setup Guide

### Option 1: Windows with WSL2 (Recommended for GPU) ✅

**Recommanded System Setup:**
- Windows 10/11 with WSL2(Ubuntu) installed
- NVIDIA GPU (e.g., RTX 4060)
- NVIDIA driver installed in Windows

#### Step 0: Verify WSL2 and GPU
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

#### Step 1: Open WSL2
```powershell
# In Windows PowerShell
wsl

# OR
wsl -d Ubuntu
```

#### Step 2: Navigate to Project in WSL2
```bash
# Open WSL2 Ubuntu terminal (or use: wsl -d Ubuntu)
# Your Windows files are at /mnt/DRIVE/...
cd /mnt/ ... /ML-BIS/MoveNet
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

#### Step 4: Create Virtual Environment (Optional)
**NOTE: It may take more time for program to setup and packages to be loaded when using virtual environment under WSL. You can ignore this step**
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

# Install system libraries
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

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
# Under MoveNet:
python3 ./utilities/check_gpu.py
```

**Expected output:**
```
============================================================
TensorFlow GPU Check
============================================================

TensorFlow version: 2.20.0
Built with CUDA: True

Physical devices:
  CPUs: 1
  GPUs: 1

GPU Details:
  GPU 0: /physical_device:GPU:0
    Device Type: GPU

Testing GPU computation...
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1766461285.048436   75740 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 5561 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.9
  ✓ GPU computation successful!
  Result shape: (2, 2)

============================================================
```

✅ **Setup Complete!** You're ready to process videos with GPU acceleration.

#### Daily Usage (WSL2)
```bash
# 1. Open WSL2 terminal
wsl -d Ubuntu

# 2. Navigate to project
cd /mnt/ ... /MoveNet_test

# 3. Activate environment (optional)
source movenet_env_wsl/bin/activate

# 4. Run your scripts
python extract_keypoints_to_2d.py annotation-dir \
    --input data/train/raw/ \
    --output data/train/transformed/
```

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

## Usage
1. Keypoints Extraction: `extract_keypoints_to_2d.py`<br>
Refer: `EXTRACT_KEYPOINTS_TO_2D_GUIDE.md`
2. Realtime Inference: `realtime_inference.py`<br>
Refer: `REALTIME_INFERENCE_GUIDE.md`