import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam

print(tf.__version__)

(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

np.set_printoptions(linewidth=200)
plt.imshow(training_images[0])

print(training_labels[0])
print(training_images[0])

training_images = training_images / 255.0
test_images = test_images / 255.0
print(training_images.shape)
print(test_images.shape)
print(training_labels.shape)
print(test_labels.shape)
print(np.unique(test_labels))
print(training_images[0].shape)


class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.1:
            print("\nReached loss less than 0.4 so cancelling training!")
            self.model.stop_training = True


model = Sequential([
                    Flatten(input_shape=(28, 28), name="input_layer"),
                    Dense(512, activation=tf.nn.relu, name="layer1"),
                    Dense(512, activation=tf.nn.relu, name="layer2"),
                    Dense(10, activation=tf.nn.softmax, name="output_layer")
])

print(model.weights)

model.compile(optimizer=Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

model.fit(training_images, training_labels, epochs=20, callbacks=myCallback())

print(model.evaluate(test_images, test_labels))

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
