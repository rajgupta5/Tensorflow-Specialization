# !wget --no-check-certificate \
#     "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip" \
#     -O "/tmp/happy-or-sad.zip"

import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

DESIRED_ACCURACY = 0.999


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > DESIRED_ACCURACY):
            print("\nReached 99.8% accuracy so cancelling training!")
            self.model.stop_training = True


# basepath = '/tmp/'
basepath = '/Users/rajkgupta/Downloads/'
local_zip = basepath + 'happy-or-sad.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(basepath + 'happy-or-sad')

# Directory with our training horse pictures
train_happy_dir = os.path.join(basepath + 'happy-or-sad/happy')

# Directory with our training human pictures
train_sad_dir = os.path.join(basepath + 'happy-or-sad/sad')

train_happy_names = os.listdir(train_happy_dir)
print(train_happy_names[:10])

train_sad_names = os.listdir(train_sad_dir)
print(train_sad_names[:10])

print('total training happy images:', len(os.listdir(train_happy_dir)))
print('total training sad images:', len(os.listdir(train_sad_dir)))

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_happy_pix = [os.path.join(train_happy_dir, fname)
                  for fname in train_happy_names[pic_index - 8:pic_index]]
next_sad_pix = [os.path.join(train_sad_dir, fname)
                for fname in train_sad_names[pic_index - 8:pic_index]]

for i, img_path in enumerate(next_happy_pix + next_sad_pix):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off')  # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1 / 255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    basepath + 'happy-or-sad/',  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=10,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

history = model.fit(train_generator, steps_per_epoch=8, epochs=15, verbose=1, callbacks=myCallback())
