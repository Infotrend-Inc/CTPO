import tensorflow as tf
from tensorflow.python.client import device_lib

print("")
print("(!! the following is build device specific, and here only to confirm hardware availability, ignore !!)")
print("--- All seen hardware    :\n", device_lib.list_local_devices())
print("--- TF GPU Available     :\n", tf.config.experimental.list_physical_devices('GPU'))

