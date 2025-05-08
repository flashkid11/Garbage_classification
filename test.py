import os
print("---- Environment Variables from Python ----")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME')}")
print(f"PATH: {os.environ.get('PATH')}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
print("-------------------------------------------")

import tensorflow as tf

print(f"TensorFlow Version: {tf.__version__}")
print(f"Is TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")

build_info = tf.sysconfig.get_build_info()
print(f"TF Build Info - CUDA Version: {build_info.get('cuda_version', 'N/A')}")
print(f"TF Build Info - cuDNN Version: {build_info.get('cudnn_version', 'N/A')}")

print("\n---- Attempting to list physical GPUs ----")
try:
    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f"Physical GPUs found: {gpu_devices}")
    if gpu_devices:
        print("SUCCESS: GPU detected by TensorFlow!")
        # More details if GPU is found
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True) # Optional, good practice
            print(f"Details for {gpu.name}: {tf.config.experimental.get_device_details(gpu)}")
    else:
        print("FAILURE: No physical GPUs detected by TensorFlow.")
except Exception as e:
    print(f"An error occurred while trying to list physical GPUs: {e}")

print("\n---- Attempting to load a CUDA library directly (for debugging) ----")
import ctypes
try:
    # Try loading a core CUDA runtime library that TF would need.
    # Adjust soname if necessary (e.g. libcudart.so.12 for CUDA 12.x)
    ctypes.CDLL("libcudart.so.12") # For CUDA 12.x. If you downgraded to CUDA 11.x, it would be libcudart.so.11.0
    print("Successfully loaded libcudart.so.12 using ctypes.")
except OSError as e:
    print(f"Failed to load libcudart.so.12 using ctypes: {e}")
    print("This indicates a problem with LD_LIBRARY_PATH or the library installation itself.")

try:
    ctypes.CDLL("libcudnn.so.8") # For cuDNN 8.x
    print("Successfully loaded libcudnn.so.8 using ctypes.")
except OSError as e:
    print(f"Failed to load libcudnn.so.8 using ctypes: {e}")
    print("This indicates a problem with LD_LIBRARY_PATH or the cuDNN library installation/symlinks.")
print("-------------------------------------------")