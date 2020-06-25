# -*- coding: utf-8 -*-
"""Device placement.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CaupoLSZHkyYghVyN8vgsPBgTEm4-hUr

# Device placement

In this reading, we are going to be looking at device placement. We will see how to access the device associated to a given tensor, and compare the use of GPUs and CPUs.

When running this notebook, ensure that the GPU runtime type is selected (Runtime -> Change runtime type).
"""

! pip install tensorflow==2.1.0
import tensorflow as tf
print(tf.__version__)

"""## Get the physical devices

First, we can list the physical devices available.
"""

# List all physical devices

tf.config.list_physical_devices()

"""If you have enabled the GPU runtime, then you should see the GPU device in the above list.

We can also check specifically for the GPU or CPU devices.
"""

# Check for GPU devices

tf.config.list_physical_devices('GPU')

# Check for CPU devices

tf.config.list_physical_devices('CPU')

"""We can get the GPU device name as follows:"""

# Get the GPU device name

tf.test.gpu_device_name()

"""## Placement of Tensor operations

TensorFlow will automatically allocate Tensor operations to a physical device, and will handle the copying between CPU and GPU memory if necessary. 

Let's define a random Tensor:
"""

# Define a Tensor

x = tf.random.uniform([3, 3])

"""We can see which device this Tensor is placed on using its `device` attribute."""

# Get the Tensor device

x.device

"""The above string will end with `'GPU:K'` if the Tensor is placed on the `K`-th GPU device. We can also check if a tensor is placed on a specific device by using `device_endswith`:"""

# Test for device allocation

print("Is the Tensor on CPU #0:  "),
print(x.device.endswith('CPU:0'))
print('')
print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))

"""## Specifying device placement

As mentioned previously, TensorFlow will automatically allocate Tensor operations to specific devices. However, it is possible to force placement on specific devices, if they are available. 

We can view the benefits of GPU acceleration by running some tests and placing the operations on the CPU or GPU respectively.
"""

# Define simple tests to time computation speed

import time

def time_matadd(x):
    start = time.time()
    for loop in range(10):
        tf.add(x, x)
    result = time.time()-start
    print("Matrix addition (10 loops): {:0.2f}ms".format(1000*result))


def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x, x)
    result = time.time()-start
    print("Matrix multiplication (10 loops): {:0.2f}ms".format(1000*result))

"""In the following cell, we run the above tests inside the context `with tf.device("CPU:0")`, which forces the operations to be run on the CPU."""

# Force execution on CPU

print("On CPU:")
with tf.device("CPU:0"):
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    time_matadd(x)
    time_matmul(x)

"""And now run the same operations on the GPU:"""

# Force execution on GPU #0 if available

if tf.config.experimental.list_physical_devices("GPU"):
    print("On GPU:")
    with tf.device("GPU:0"): 
        x = tf.random.uniform([1000, 1000])
        assert x.device.endswith("GPU:0")
        time_matadd(x)
        time_matmul(x)

"""Note the significant time difference between running these operations on different devices.

## Model training

Finally, we will demonstrate that GPU device placement offers speedup benefits for model training.
"""

# Load the MNIST dataset

from tensorflow.keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255., x_test/255.

# Reduce the dataset size to speed up the test

x_train, y_train = x_train[:1000], y_train[:1000]

# Define a function to build the model

from tensorflow.keras import layers
from tensorflow.keras.models import  Sequential

def get_model():
  model = Sequential([
      layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
      layers.MaxPooling2D((2, 2)),
      layers.Flatten(),
      layers.Dense(64, activation='relu'),
      layers.Dense(10, activation='softmax'),
      ])
  return model

# Time a training run on the CPU

with tf.device("CPU:0"):
  model = get_model()
  model.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  start = time.time()
  model.fit(x_train[..., np.newaxis], y_train, epochs=5, verbose=0)
  result = time.time() - start

print("CPU training time: {:0.2f}ms".format(1000 * result))

# Time a training run on the GPU

with tf.device("GPU:0"):
  model = get_model()
  model.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  start = time.time()
  model.fit(x_train[..., np.newaxis], y_train, epochs=5, verbose=0)
  result = time.time() - start

print("GPU training time: {:0.2f}ms".format(1000 * result))

"""## Further reading and resources 
* https://www.tensorflow.org/tutorials/customization/basics#gpu_acceleration
"""