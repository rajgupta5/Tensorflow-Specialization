from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, AveragePooling2D, Flatten
import numpy as np
import tensorflow as tf

#
#
# model = Sequential([
#     Dense(1, input_shape=(4,))
# ])
# kernel, bias = model.weights
# print(kernel)
# print(bias)
# print(kernel.numpy())
# print(bias.numpy())
#
#
#
# inputs = Input(shape=(16, 16, 3))
# h = Conv2D(32, 3, activation='relu')(inputs)
# h = AveragePooling2D(3)(h)
# outputs = Flatten()(h)
# Model = Model(inputs=inputs, outputs=outputs)
# print(Model.summary())
# print(outputs)


# Create Variable objects of different type with tf.Variable
strings = tf.Variable(["Hello world!"], tf.string)
floats  = tf.Variable([3.14159, 2.71828], tf.float64)
ints = tf.Variable([1, 2, 3], tf.int32)
complexs = tf.Variable([25.9 - 7.39j, 1.23 - 4.91j], tf.complex128)

print(tf.Variable(tf.constant(4.2, shape=(3,3))))

# Use the value of a Variable
v = tf.Variable(0.0)
w = v + 1  # w is a tf.Tensor which is computed based on the value of v.
print(type(w))

v.assign_add(1)
print(v)
v.assign_sub(1)
print(v)


# Create a constant Tensor
x = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x)
print("dtype:", x.dtype)
print("shape:", x.shape)

print(x.numpy())

x = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
print(x)


coeffs = np.arange(16)
shape1 = [8,2]
shape2 = [4,4]
shape3 = [2,2,2,2]


# Create Tensors of different shape
a = tf.constant(coeffs, shape=shape1)
print("\n a:\n ", a)
b = tf.constant(coeffs, shape=shape2)
print("\n b:\n ", b)
c = tf.constant(coeffs, shape=shape3)
print("\n c:\n ", c)


# Create a constant Tensor
t = tf.constant(np.arange(80), shape=[5,2,8])
rank = tf.rank(t)
# Display the rank
print("rank: ", rank)


t2 = tf.reshape(t, [8,10])
# Display the new shape
print("t2.shape: ", t2.shape)

ones = tf.ones(shape=(2,3))
zeros = tf.zeros(shape=(2,4))
eye = tf.eye(3)
tensor7 = tf.constant(7.0, shape=[2,2])

# Display the created tensors
print("\n Ones:\n ", ones)
print("\n Zeros:\n ", zeros)
print("\n Identity:\n ", eye)
print("\n Tensor filled with 7: ", tensor7)


# Create a ones Tensor and a zeros Tensor
t1 = tf.ones(shape=(2, 2))
t2 = tf.zeros(shape=(2, 2))

concat0 = tf.concat([t1, t2], 0)
concat1 = tf.concat([t1, t2], 1)

# Display the concatenated tensors
print(concat0)
print(concat1)


# Create a constant Tensor
t = tf.constant(np.arange(24), shape=(3, 2, 4))
print("\n t shape: ", t.shape)

t1 = tf.expand_dims(t,0)
t2 = tf.expand_dims(t,1)
t3 = tf.expand_dims(t,2)

# Display the shapes after tf.expand_dims
print("\n After expanding dims:\n t1 shape: ", t1.shape, "\n t2 shape: ", t2.shape, "\n t3 shape: ", t3.shape)

t1 = tf.squeeze(t1,0)
t2 = tf.squeeze(t2,1)
t3 = tf.squeeze(t3,2)

# Display the shapes after tf.squeeze
print("\n After squeezing:\n t1 shape: ", t1.shape, "\n t2 shape: ", t2.shape, "\n t3 shape: ", t3.shape)

x = tf.constant([1,2,3,4,5,6,7])
print(x[1:-3])

# Create two constant Tensors
c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
d = tf.constant([[1.0, 1.0], [0.0, 1.0]])

matmul_cd = tf.matmul(c,d)

# Display the result
print("\n tf.matmul(c,d):\n", matmul_cd)

c_times_d = c*d
c_plus_d = c+d
c_minus_d = c-d
c_div_c = c/c
# Display the results
print("\n c*d:\n", c_times_d)
print("\n c+d:\n", c_plus_d)
print("\n c-d:\n", c_minus_d)
print("\n c/c:\n", c_div_c)


# Create Tensors
a = tf.constant([[2, 3], [3, 3]])
b = tf.constant([[8, 7], [2, 3]])
x = tf.constant([[-6.89 + 1.78j], [-2.54 + 2.15j]])

absx = tf.abs(x)
powab = tf.pow(a,b)

# Display the results
print("\n ", absx)
print("\n ", powab)


tn = tf.random.normal(shape=(2,2), mean=0, stddev=1.)
tu = tf.random.uniform(shape=(2,1), minval=0, maxval=10, dtype='int32')
tp = tf.random.poisson((2,2), 5)

# More maths operations
d = tf.square(tn)
e = tf.exp(d)
f = tf.cos(c)