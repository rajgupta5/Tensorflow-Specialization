import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam


class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.998):
            print("\nReached 99.8% accuracy so cancelling training!")
            self.model.stop_training = True


(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

np.set_printoptions(linewidth=200)
plt.imshow(training_images[0])

training_images = training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

training_images = training_images / 255.0
test_images = test_images / 255.0

print(training_images.shape)
print(test_images.shape)
print(training_labels.shape)
print(test_labels.shape)
print(np.unique(test_labels))

model = Sequential([
    Conv2D(32, kernel_size=3, padding='VALID', activation='relu', input_shape=(28, 28, 1), data_format='channels_last'),
    MaxPooling2D(pool_size=2),
    # Conv2D(64, (3, 3), activation='relu'),
    # MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation=tf.nn.relu),
    # Dense(512, activation=tf.nn.relu),
    Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

history = model.fit(training_images, training_labels, epochs=10, callbacks=myCallback(), validation_split=0.2)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)


# -----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
# -----------------------------------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))  # Get number of epochs

# ------------------------------------------------
# Plot training and validation accuracy per epoch
# ------------------------------------------------
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.figure()

# ------------------------------------------------
# Plot training and validation loss per epoch
# ------------------------------------------------
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')

print(tf.__version__)