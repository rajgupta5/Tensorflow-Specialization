def _short(channel_in, channel_out):
    print(channel_in)
    print(channel_out)

    if channel_in != channel_out:
        return Conv2D(channel_out, kernel_size=(1, 1), padding='same', name='ResidualBlock_Conv4')
    else:
        return lambda x: x

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

input_shape=(28, 28, 1)
channel = input_shape[1]

inputs = Input(shape=input_shape)
h = BatchNormalization( name='ResidualBlock_bn1')(inputs)
h = tf.nn.relu(h)
h = Conv2D(channel, kernel_size=(3, 3), padding='same', name='ResidualBlock_Conv2')(h)
h = BatchNormalization(name='ResidualBlock_bn2')(h)
h = tf.nn.relu(h)
h = Conv2D(channel, kernel_size=(3, 3), padding='same', name='ResidualBlock_Conv3')(h)
print(h)
sc = _short(inputs.shape[-1], h.get_shape()[-1])

print(h, sc)
outputs = tf.keras.layers.Add()()([h, sc])
model = Model(inputs=inputs, outputs=outputs)


# input1 = tf.keras.layers.Input(shape=(16,))
# x1 = tf.keras.layers.Dense(8, activation='relu')(input1)
# input2 = tf.keras.layers.Input(shape=(32,))
# x2 = tf.keras.layers.Dense(8, activation='relu')(input2)
# # equivalent to `added = tf.keras.layers.add([x1, x2])`
# added = tf.keras.layers.Add()([x1, x2])
# out = tf.keras.layers.Dense(4)(added)
# model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)