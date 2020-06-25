# packages
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import tensorflow_hub as hub
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import mnist
import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import json
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50

print(tf.__version__)

# callbacks
checkpoint_best_path = 'model_checkpoint_best/checkpoint'
checkpoint_best = ModelCheckpoint(filepath=checkpoint_best_path, save_weights_only=True, save_freq='epoch', \
                                  monitor='val_accuracy', save_best_only=True, verbose=1)

early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, delta=0.01, mode='max')



# custom callback
class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


# load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# train and test normalization
x_train = x_train / 255.0
x_test = x_test / 255.0

# image data generator
# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1 / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1 / 255)

train_generator = train_datagen.flow_from_directory(
    basepath + 'horse-or-human/',  # This is the source directory for training images
    target_size=(300, 300),  # All images will be resized to 150x150
    batch_size=128,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    basepath + 'validation-horse-or-human/',  # This is the source directory for training images
    target_size=(300, 300),  # All images will be resized to 150x150
    batch_size=32,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

# Sequenatial API
# activation - 'sigmoid', 'relu', 'softmax', tf.nn.relu, tf.nn.softmax
model = Sequential([
    Conv2D(filters=16, input_shape=(32, 32, 3), kernel_size=(3, 3),
           activation='relu', name='conv_1'),
    Conv2D(filters=8, kernel_size=(3, 3), activation='relu', name='conv_2'),
    MaxPooling2D(pool_size=(4, 4), name='pool_1'),
    Dropout(0.5),
    Flatten(name='flatten'),
    Dense(units=32, activation='relu', name='dense_1'),
    Dense(units=10, activation='softmax', name='dense_2')
])


#Functional API





# model compile
# optimizer - 'adam', RMSprop(lr=0.001), 'rmsprop'
# loss - sparse_categorical_crossentropy, binary_crossentropy, Huber
# metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# model summary
print(model.summary())

# model.fit
# validation_data=validation_generator
# verbose - 0,1,2
# train_generator
history = model.fit(x=x_train, y=y_train, epochs=50, validation_data=(x_test, y_test), batch_size=10, steps_per_epoch=8,
                    validation_steps=8,
                    callbacks=[checkpoint_best], verbose=0)

# model evaluate
model.evaluate(x_test, y_test)

# model predict
model.predict(x_samples)

# keras hdf5 model save
model.save("my_model.h5")
# Savedmodel format
model.save('my_model')

# load model
new_model = load_model('my_model')
new_model1 = load_model('my_model.h5')

# Load weights -- accuracy is the same as the trained model
model.load_weights(checkpoint_path)

# loading pre trained models like resnet50
model = ResNet50(weights='imagenet', include_top=True)
resnet_model = load_model('models/Keras_Resnet50.h5')

from tensorflow.keras.preprocessing.image import load_img
lemon_img = load_img('lemon.jpg', target_size=(224, 224))

def get_top_5_predictions(img):
    x = img_to_array(img)[np.newaxis, ...]
    x = preprocess_input(x)
    preds = decode_predictions(model.predict(x), top=5)
    top_preds = pd.DataFrame(columns=['prediction', 'probability'],
                             index=np.arange(5)+1)
    for i in range(5):
        top_preds.loc[i+1, 'prediction'] = preds[0][i][1]
        top_preds.loc[i+1, 'probability'] = preds[0][i][2]
    return top_preds

# using models from tensorflow hub
# Build Google's Mobilenet v1 model
module_url = "https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/4"
model = Sequential([hub.KerasLayer(module_url)])
model.build(input_shape=[None, 160, 160, 3])

