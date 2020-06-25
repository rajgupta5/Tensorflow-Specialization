#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
print(tf.__version__)


# # Validation, regularisation and callbacks

#  ## Coding tutorials
#  #### [1. Validation sets](#coding_tutorial_1)
#  #### [2. Model regularisation](#coding_tutorial_2)
#  #### [3. Introduction to callbacks](#coding_tutorial_3)
#  #### [4. Early stopping / patience](#coding_tutorial_4)

# ***
# <a id="coding_tutorial_1"></a>
# ## Validation sets

# #### Load the data

# In[3]:


# Load the diabetes dataset

from sklearn.datasets import load_diabetes
diabetes_dataset = load_diabetes()
print(diabetes_dataset['DESCR'])


# In[7]:


# Save the input and target variables
diabetes_dataset.keys()
data = diabetes_dataset.data
target = diabetes_dataset.target
target[:5]


# In[8]:


# Normalise the target data (this will make clearer training curves)

target = (target - target.mean(axis=0))/target.std()
target[:5]


# In[12]:


# Split the data into train and test sets

from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.1)
print(train_data.shape)
print(test_data.shape)
print(train_target.shape)
print(test_target.shape)


# #### Train a feedforward neural network model

# In[ ]:


# Build the model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def get_model():
  model = Sequential([
                      Dense(128, activation='relu', input_shape=(train_data.shape[1],)),
                      Dense(128, activation='relu'),
                      Dense(128, activation='relu'),
                      Dense(128, activation='relu'),
                      Dense(128, activation='relu'),
                      Dense(128, activation='relu'),
                      Dense(1)
  ])
  return model

model = get_model() 


# In[117]:


# Print the model summary
model.summary()


# In[ ]:


# Compile the model
model.compile(optimizer='adam', loss = 'mse', metrics = ['mse'])


# In[119]:


# Train the model, with some of the data reserved for validation
history = model.fit(train_data, train_target, epochs=100, verbose=True, validation_split=0.3, batch_size=64)


# In[120]:


# Evaluate the model on the test set

model.evaluate(test_data, test_target)


# #### Plot the learning curves

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[122]:


# Plot the training and validation loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()


# ***
# <a id="coding_tutorial_2"></a>
# ## Model regularisation

# #### Adding regularisation with weight decay and dropout

# In[ ]:


from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers


# In[ ]:


def get_regularised_model(wd, rate):
    model = Sequential([
        Dense(128, activation="relu", input_shape=(train_data.shape[1],), kernel_regularizer=tf.keras.regularizers.l1_l2(wd)),
        Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(wd)),
        Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(wd)),
        Dropout(rate),
        Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(wd)),
        Dropout(rate),
        Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(wd)),
        Dropout(rate),
        Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(wd)),
        Dropout(rate),
        Dense(1)
    ])
    return model


# In[151]:


# Re-build the model with weight decay and dropout layers
model = get_regularised_model(1e-5, 0.3)
model.summary()


# In[ ]:


# Compile the model
model.compile(optimizer='adam', loss = 'mse', metrics = ['mse'])



# In[153]:


# Train the model, with some of the data reserved for validation
history = model.fit(train_data, train_target, epochs=100, verbose=True, validation_split=0.25, batch_size=64)



# In[154]:


# Evaluate the model on the test set
model.evaluate(test_data, test_target)


# #### Plot the learning curves

# In[ ]:


# Plot the training and validation loss

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()


# ***
# <a id="coding_tutorial_3"></a>
# ## Introduction to callbacks

# #### Example training callback

# In[ ]:


def on_train_begin(self, logs=None):
      # Do something

    def on_train_batch_begin(self, batch, logs=None):
      # DO Something

    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.1:
            print("\nReached loss less than 0.4 so cancelling training!")
            self.model.stop_training = True


# In[ ]:


# Write a custom callback
from tensorflow.keras.callbacks import Callback

class TrainingCallback(Callback):
  def on_train_begin(self, logs=None):
    print("String training...")
  def on_epoch_begin(self, epoch, logs=None):
    print(f"Starting epoch {epoch} ")
  def on_train_batch_begin(self, batch, logs=None):
    print(f"Staring batch {batch}")
  def on_train_batch_end(self, batch, logs=None):
    print(f"Ending batch {batch}")
  def on_epoch_end(self, epoch, logs=None):
    print(f"Ending epoch {epoch} ")
  def on_train_end(self, logs=None):
    print("Ending training...")


class TestingCallback(Callback):
  def on_test_begin(self, logs=None):
    print("String testing...")
  def on_test_batch_begin(self, batch, logs=None):
    print(f"Staring batch {batch}")
  def on_test_batch_end(self, batch, logs=None):
    print(f"Ending batch {batch}")
  def on_test_end(self, logs=None):
    print("Ending testing...")

class PredictionCallback(Callback):
  def on_predict_begin(self, logs=None):
    print("String Predicting...")
  def on_predict_batch_begin(self, batch, logs=None):
    print(f"Staring batch {batch}")
  def on_predict_batch_end(self, batch, logs=None):
    print(f"Ending batch {batch}")
  def on_predict_end(self, logs=None):
    print("Ending Predicting...")


# In[ ]:


# Re-build the model
model = get_regularised_model(1e-5, 0.3)


# In[ ]:


# Compile the model
model.compile(optimizer='adam', loss = 'mse')



# #### Train the model with the callback

# In[166]:


# Train the model, with some of the data reserved for validation

history = model.fit(train_data, train_target, epochs=3, verbose=True, validation_split=0.25, batch_size=64, callbacks=TrainingCallback())


# In[170]:


# Evaluate the model
model.evaluate(test_data, test_target, callbacks=TestingCallback())



# In[177]:


# Make predictions with the model

model.predict(test_data, verbose = 2, callbacks = PredictionCallback())


# ***
# <a id="coding_tutorial_4"></a>
# ## Early stopping / patience

# #### Re-train the models with early stopping

# In[197]:


# Re-train the unregularised model

from tensorflow.keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, delta=0.01, mode='max')

unregularized_model = get_model()
unregularized_model.compile(optimizer='adam', loss = 'mse')
unreg_history = unregularized_model.fit(train_data, train_target, epochs=100, verbose=True, validation_split=0.15, batch_size=64, callbacks=[EarlyStopping(patience=2)])





# In[198]:


# Evaluate the model on the test set
unregularized_model.evaluate(test_data, test_target, verbose=2)


# In[199]:


# Re-train the regularised model
regularized_model = get_regularised_model(1e-8, 0.2)
regularized_model.compile(optimizer='adam', loss = 'mse')
reg_history = regularized_model.fit(train_data, train_target, epochs=100, verbose=True, validation_split=0.15, batch_size=64, callbacks=[EarlyStopping(patience=2)])





# In[200]:


# Evaluate the model on the test set
regularized_model.evaluate(test_data, test_target, verbose=2)



# #### Plot the learning curves

# In[201]:


# Plot the training and validation loss

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 5))

fig.add_subplot(121)

plt.plot(unreg_history.history['loss'])
plt.plot(unreg_history.history['val_loss'])
plt.title('Unregularised model: loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

fig.add_subplot(122)

plt.plot(reg_history.history['loss'])
plt.plot(reg_history.history['val_loss'])
plt.title('Regularised model: loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.show()


# In[ ]:




