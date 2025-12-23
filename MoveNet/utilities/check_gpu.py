"""
Quick script to check if TensorFlow can detect and use GPU.
"""

import tensorflow as tf

print("=" * 60)
print("TensorFlow GPU Check")
print("=" * 60)

# Check TensorFlow version
print(f"\nTensorFlow version: {tf.__version__}")

# Check if built with CUDA
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

# List physical devices
print("\nPhysical devices:")
print(f"  CPUs: {len(tf.config.list_physical_devices('CPU'))}")
print(f"  GPUs: {len(tf.config.list_physical_devices('GPU'))}")

# Detailed GPU info
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("\nGPU Details:")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
        print(f"    Device Type: {gpu.device_type}")
    
    # Test GPU computation
    print("\nTesting GPU computation...")
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
        print("  ✓ GPU computation successful!")
        print(f"  Result shape: {c.shape}")
    except Exception as e:
        print(f"  ✗ GPU computation failed: {e}")
else:
    print("\n⚠ No GPU detected!")
    print("\nTo use GPU, you need:")
    print("  1. NVIDIA GPU with CUDA Compute Capability 3.5+")
    print("  2. CUDA Toolkit installed")
    print("  3. cuDNN library installed")
    print("  4. TensorFlow GPU version: pip install tensorflow[and-cuda]")

print("\n" + "=" * 60)
