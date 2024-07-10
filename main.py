import tensorflow as tf

print("TensorFlow Version: ", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# List all physical devices
physical_devices = tf.config.list_physical_devices('GPU')
print("Physical GPUs:", physical_devices)

# Check if TensorFlow can use the GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())