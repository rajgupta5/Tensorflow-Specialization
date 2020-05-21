import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import Callback


class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


(x_train, y_train), (x_test, y_test) = mnist.load_data()
plt.imshow(x_train[0])

print(y_train[0])
print(x_train[0])

x_train = x_train / 255.0
x_test = x_test / 255.0

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(np.unique(y_test))

model = Sequential([
                    Flatten(input_shape=(28, 28)),
                    Dense(512, activation=tf.nn.relu),
                    Dense(512, activation=tf.nn.relu),
                    Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train, epochs=20, callbacks=myCallback())

print(model.evaluate(x_test, y_test))

classifications = model.predict(x_test)
print(classifications[0])
print(y_test[0])