#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
tf.__version__


# # Introduction to TensorFlow 2
# 
# ## Coding tutorials
# #### [1. Hello TensorFlow!](#coding_tutorial_1)

# ---
# <a id='coding_tutorial_1'></a>
# ## Hello TensorFlow!

# In[ ]:


# Import TensorFlow
import tensorflow as tf



# In[2]:


# Check its version

tf.__version__


# In[3]:


# Train a feedforward neural network for image classification

print('Loading data...\n')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print('MNIST dataset loaded.\n')

x_train = x_train/255.

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print('Training model...\n')
model.fit(x_train, y_train, epochs=3, batch_size=32)

print('Model trained successfully!')

