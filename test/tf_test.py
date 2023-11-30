import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import timeit

#tf.debugging.set_log_device_placement(True)

def matmul_test(device, devname):
  with tf.device(device):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
    print(f"On {devname}:")
    print(c)

device_name = tf.test.gpu_device_name()
have_gpu = False
if device_name == '/device:GPU:0':
  have_gpu = True

if have_gpu:
  print("Tensorflow test: GPU found")
else:
  print("Tensorflow test: CPU only")

print("\n\n")

matmul_test('/CPU:0', 'CPU')
if have_gpu:
  matmul_test('/GPU:0', 'GPU')

print("\n\n")

# Adapted from https://colab.research.google.com/notebooks/gpu.ipynb
def cpu():
  with tf.device('/cpu:0'):
    random_image_cpu = tf.random.normal((100, 100, 100, 3))
    net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
    return tf.math.reduce_sum(net_cpu)

def gpu():
  with tf.device('/device:GPU:0'):
    random_image_gpu = tf.random.normal((100, 100, 100, 3))
    net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
    return tf.math.reduce_sum(net_gpu)
  
# We run each op once to warm up; see: https://stackoverflow.com/a/45067900
cpu()
if have_gpu:
  gpu()

# Run the op several times.
print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
      '(batch x height x width x channel). Sum of ten runs.')

cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(f"CPU (s): {cpu_time}")

if have_gpu:
  gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
  print(f"GPU (s): {gpu_time}")

  print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))

print("Tensorflow test: Done")
