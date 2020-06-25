
import tensorflow as tf
print("TensorFlow version: {}".format(tf.__version__))
print("Keras version: {}".format(tf.keras.__version__))


variable = tf.Variable([3, 3])
if tf.test.is_gpu_available():
    print('GPU')
    print('GPU #0?')
    print(var.device.endswith('GPU:0'))
else:
    print('CPU')


tf.config.list_physical_devices('CPU')


# ### Tensor Constant

# In[3]:


ineuron = tf.constant(42)
ineuron


# In[16]:


ineuron.numpy()


# In[17]:


ineuron1 = tf.constant(1, dtype = tf.int64)
ineuron1


# In[18]:


ineuron_x = tf.constant([[4,2],[9,5]])
print(ineuron_x)


# In[19]:


ineuron_x.numpy()


# In[20]:


print('shape:',ineuron_x.shape)
print(ineuron_x.dtype)


# #### Commonly used method is to generate constant tf.ones and the tf.zeros like of numpy np.ones & np.zeros

# In[21]:


print(tf.ones(shape=(2,3)))


# In[22]:


print(tf.zeros(shape=(3,2)))


# In[23]:


import tensorflow as tf
const2 = tf.constant([[3,4,5], [3,4,5]]);tf
const1 = tf.constant([[1,2,3], [1,2,3]]);
result = tf.add(const1, const2);
print(result)


# >We have defined two constants and we add one value to the other. 
# >As a result, we got a Tensor object with the result of the adding. 

# #### Random constant

# In[24]:


tf.random.normal(shape=(2,2),mean=0,stddev=1.0)


# In[25]:


tf.random.uniform(shape=(2,2),minval=0,maxval=10,dtype=tf.int32)


# In[ ]:





# ### Variables

# >A variable is a special tensor that is used to store variable values ​​and needs to be initialized with some values

# #### Declaring variables

# In[26]:


var0 = 24 # python variable
var1 = tf.Variable(42) # rank 0 tensor
var2 = tf.Variable([ [ [0., 1., 2.], [3., 4., 5.] ], [ [6., 7., 8.], [9., 10., 11.] ] ]) #rank 3 tensor
var0, var1, var2


# >TensorFlow will infer the datatype, defaulting to tf.float32 for floats and tf.int32 for integers

# #### The datatype can be explicitly specified

# In[27]:


float_var64 = tf.Variable(89, dtype = tf.float64)
float_var64.dtype


# `TensorFlow has a large number of built-in datatypes.`
# * tf.float16: 16-bit half-precision floating-point.
# * tf.float32: 32-bit single-precision floating-point.
# * tf.float64: 64-bit double-precision floating-point.
# * tf.bfloat16: 16-bit truncated floating-point.
# * tf.complex64: 64-bit single-precision complex.
# * tf.complex128: 128-bit double-precision complex.
# * tf.int8: 8-bit signed integer.
# * tf.uint8: 8-bit unsigned integer.
# * tf.uint16: 16-bit unsigned integer.
# * tf.uint32: 32-bit unsigned integer.
# * tf.uint64: 64-bit unsigned integer.
# * tf.int16: 16-bit signed integer.
# * tf.int32: 32-bit signed integer.
# * tf.int64: 64-bit signed integer.
# * tf.bool: Boolean.
# * tf.string: String.
# * tf.qint8: Quantized 8-bit signed integer.
# * tf.quint8: Quantized 8-bit unsigned integer.
# * tf.qint16: Quantized 16-bit signed integer.
# * tf.quint16: Quantized 16-bit unsigned integer.
# * tf.qint32: Quantized 32-bit signed integer.
# * tf.resource: Handle to a mutable resource.
# * tf.variant: Values of arbitrary types.
# 

# #### To reassign a variable, use var.assign()

# In[28]:


var_reassign = tf.Variable(89.)
var_reassign


# In[29]:


var_reassign.assign(98.)
var_reassign


# In[30]:


initial_value = tf.random.normal(shape=(2,2))
a = tf.Variable(initial_value)
print(a)


# >We can assign "=" with assign (value), or assign_add (value) with "+ =", or assign_sub (value) with "-="

# In[31]:


new_value = tf.random.normal(shape=(2, 2))
a.assign(new_value)
for i in range(2):
    for j in range(2):
        assert a[i, j] == new_value[i, j]


# In[32]:


added_value = tf.random.normal(shape=(2,2))
a.assign_add(added_value)
for i in range(2):
    for j in range(2):
        assert a[i,j] == new_value[i,j]+added_value[i,j]


# #### Shaping a tensor

# In[33]:


tensor = tf.Variable([ [ [10., 11., 12.], [13., 14., 15.] ], [ [16., 17., 18.], [19., 20., 21.] ] ]) # tensor variable
print(tensor.shape)


# #### Tensors can be reshaped and retain the same values which is required for constructing Neural networks.

# In[34]:


tensor1 = tf.reshape(tensor,[2,6]) # 2 rows 6 cols
#tensor2 = tf.reshape(tensor,[1,12]) # 1 rows 12 cols
tensor1


# In[35]:


tensor2 = tf.reshape(tensor,[1,12]) # 1 row 12 columns
tensor2


# ### Rank of a tensor

# >The rank of a tensor is defined as the number of dimensions, which is the number of indices that are required to specify any particular element of that tensor.

# In[36]:


tf.rank(tensor)


# >(the shape is () because the output here is a scalar value)

# #### Specifying an element of a tensor

# In[37]:


tensor3 = tensor[1, 0, 2] # slice 1, row 0, column 2
tensor3


# #### Casting a tensor to a NumPy variable

# In[38]:


print(tensor.numpy())


# In[39]:


print(tensor[1, 0, 2].numpy())


# #### Finding the size or length of a tensor

# In[40]:


tensor_size = tf.size(input=tensor).numpy()
tensor_size


# In[41]:


#the datatype of a tensor
tensor3.dtype


# ### Tensorflow mathematical operations
# >Can be used as numpy for artificial operations. Tensorflow can not execute these operations on the GPU or TPU.
# 

# In[42]:


a = tf.random.normal(shape=(2,2))
b = tf.random.normal(shape=(2,2))
c = a + b
d = tf.square(c)
e = tf.exp(c)
print(a)
print(b)
print(c)
print(d)
print(e)


# ### Performing element-wise primitive tensor operations

# In[43]:


tensor*tensor


# ### Broadcasting in Tensorflow
# 
# >Element-wise tensor operations support broadcasting in the same way that NumPy arrays do.
# 
# >The simplest example is multiplication of  a tensor by a scalar value.

# In[44]:


tensor4 = tensor*4
print(tensor4)


# ### Transpose Matrix multiplication

# In[45]:


matrix_u = tf.constant([[6,7,6]])
matrix_v = tf.constant([[3,4,3]])
tf.matmul(matrix_u, tf.transpose(a=matrix_v))


# ### Casting a tensor to another datatype

# In[46]:


i = tf.cast(tensor1, dtype=tf.int32)
i


# ##### Casting with truncation

# In[47]:


j = tf.cast(tf.constant(4.9), dtype=tf.int32)
j


# ###  Ragged tensors

# `A ragged tensor is a tensor having one or more ragged dimensions. Ragged dimensions are dimensions that have slices having various lengths.There are a variety of methods for the declaration of ragged arrays, the simplest way is declaring a constant ragged array.`

# #### Below example shows how to declare a constant ragged array

# In[37]:


ragged =tf.ragged.constant([[9, 7, 4, 3], [], [11, 12, 8], [3], [7,8]])
print(ragged)
print(ragged[0,:])
print(ragged[1,:])
print(ragged[2,:])
print(ragged[3,:])
print(ragged[4,:])


# ### Squared difference of tensors

# In[38]:


varx = [4,5,6,1,2]
vary = 8
varz = tf.math.squared_difference(varx,vary)
varz


# #### Calculate the mean

# >Function available
# >tf.reduce_mean().

# `Similar to np.mean, except that it infers the return datatype from the input tensor, whereas np.mean allows you to specify the output type`
# 
# `tf.reduce_mean(input_tensor, axis=None, keepdims=None, name=None)`

# In[39]:


# Defining a constant
numbers = tf.constant([[8., 9.], [1., 2.]])


# #### Calculate the mean across all axes

# In[40]:


tf.reduce_mean(input_tensor=numbers) #default axis = None


# #### Calculate the mean across columns (reduce rows) with this:

# In[41]:


tf.reduce_mean(input_tensor=numbers, axis=0)


# #### When keepdims = True

# In[42]:


tf.reduce_mean(input_tensor=numbers, axis=0, keepdims=True) #the reduced axis is retained with a length of 1


# #### Calculate the mean across rows (reduce columns) with this:

# In[43]:


tf.reduce_mean(input_tensor=numbers, axis=1)


# #### When keepdims= True

# In[44]:


tf.reduce_mean(input_tensor=numbers, axis=1, keepdims=True) #the reduced axis is retained with a length of 1


# ####  Random values generation

# ##### tf.random.normal()
# 
# >tf.random.normal() outputs a tensor of the given shape filled with values of the dtype type from a normal distribution.
# 
# >The function is as follows:
#     
# `tf.random.normal(shape, mean = 0, stddev =2, dtype=tf.float32, seed=None, name=None)`

# In[3]:


tf.random.normal(shape = (3,2), mean=10, stddev=2, dtype=tf.float32, seed=None, name=None)
randon_num = tf.random.normal(shape = (3,2), mean=10.0, stddev=2.0)
print(randon_num)


# ####  tf.random.uniform()

# >The function is this:
#     
# >tf.random.uniform(shape, minval = 0, maxval= None, dtype=tf.float32, seed=None, name=None)

# `This outputs a tensor of the given shape filled with values from a uniform distribution in the range minval to maxval, where the lower bound is inclusive but the upper bound isn't.
# 
# Example:`

# In[32]:


tf.random.uniform(shape = (2,4), minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)


# #### Setting the seed

# In[4]:


tf.random.set_seed(11)
random_num1 = tf.random.uniform(shape = (2,2), maxval=10, dtype = tf.int32)
random_num2 = tf.random.uniform(shape = (2,2), maxval=10, dtype = tf.int32)
print(random_num1) #Call 1
print(random_num2)


# In[5]:


tf.random.set_seed(11) #same seed
random_num1 = tf.random.uniform(shape = (2,2), maxval=10, dtype = tf.int32)
random_num2 = tf.random.uniform(shape = (2,2), maxval=10, dtype = tf.int32)
print(random_num1) #Call 2
print(random_num2)


# #### Practical example of Random values using Dices

# In[46]:


dice11 = tf.Variable(tf.random.uniform([10, 1], minval=1, maxval=7, dtype=tf.int32))
dice12 = tf.Variable(tf.random.uniform([10, 1], minval=1, maxval=7, dtype=tf.int32))

# lets ADD
dice_sum1 = dice11 + dice12
# We've got three separate 10x1 matrices. To produce a single 10x3 matrix, we'll concatenate them along dimension 1.
finale_matrix = tf.concat(values=[dice11, dice12, dice_sum1], axis=1)
print(finale_matrix)


# #### Finding the indices of the largest and smallest element

# >The following functions are available:
#     
# >`tf.argmax(input, axis=None, name=None, output_type=tf.int64 )`
# 
# >`tf.argmin(input, axis=None, name=None, output_type=tf.int64 )`

# In[47]:


# 1-D tensor
tensor_1d = tf.constant([12, 11, 51, 42, 6, 16, -8, -19, 31])
print(tensor_1d)


i = tf.argmax(input=tensor_1d)
print('index of max; ', i)
print('Max element: ',tensor_1d[i].numpy())


i = tf.argmin(input=tensor_1d,axis=0).numpy()
print('index of min: ', i)
print('Min element: ',tensor_1d[i].numpy())


# #### Saving and restoring using a checkpoint

# In[48]:


variable1 = tf.Variable([[5,6,9,3],[14,15,16,18]])
checkpoint= tf.train.Checkpoint(var=variable1)
savepath = checkpoint.save('./vars')
variable1.assign([[0,0,0,0],[0,0,0,0]])
variable1
checkpoint.restore(savepath)
print(variable1)


# #### Using tf.function
# 
# `tf.function is a function that will take a Python function and return a TensorFlow graph. The advantage of this is that graphs can apply optimizations and exploit parallelism in the Python function (func). tf.function is new to TensorFlow 2.`
# 

# >Its function is as follows:
#     
# `tf.function(
# func=None,input_signature=None,autograph=True,experimental_autograph_options=None
# )`
# 

# In[6]:


def f1(x, y):
    return tf.reduce_mean(input_tensor=tf.multiply(x ** 3, 6) + y**3)
func = tf.function(f1)
x = tf.constant([3., -4.])
y = tf.constant([1., 4.])
# f1 and f2 return the same value, but f2 executes as a TensorFlow graph
assert f1(x,y).numpy() == func(x,y).numpy()
#The assert passes, so there is no output


# ## Calculate the gradient

# ### GradientTape

# >Another difference from numpy is that it can automatically track the gradient of any variable.
# 
# >Open one GradientTape and `tape.watch()` track variables through

# In[50]:


a = tf.random.normal(shape=(2,2))
b = tf.random.normal(shape=(2,2))
with tf.GradientTape() as tape:
    tape.watch(a)
    c = tf.sqrt(tf.square(a)+tf.square(b))
    dc_da = tape.gradient(c,a)
    print(dc_da)


# >For all variables, the calculation is tracked by default and used to find the gradient, so do not `usetape.watch()`

# In[51]:


a = tf.Variable(a)
with tf.GradientTape() as tape:
    c = tf.sqrt(tf.square(a)+tf.square(b))
    dc_da = tape.gradient(c,a)
    print(dc_da)


# > You can GradientTapefind higher-order derivatives by opening a few more:

# In[52]:


with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as tape:
        c = tf.sqrt(tf.square(a)+tf.square(b))
        dc_da = tape.gradient(c,a)
    d2c_d2a = outer_tape.gradient(dc_da,a)
    print(d2c_d2a)


# # Keras, a High-Level API for TensorFlow 2

# ![title](img/keras.png)

# ## The Keras Sequential model
# 
# `To build a Keras Sequential model, you add layers to it in the same order that you want the computations to be undertaken by the network.`
# 
# `After you have built your model, you compile it; this optimizes the computations that are to be undertaken, and is where you allocate the optimizer and the loss function you want your model to use.`
# 
# `The next stage is to fit the model to the data. This is commonly known as training the model, and is where all the computations take place. It is possible to present the data to the model either in batches, or all at once.`
# 
# `Next, you evaluate your model to establish its accuracy, loss, and other metrics. Finally, having trained your model, you can use it to make predictions on new data. So, the workflow is: build, compile, fit, evaluate, make predictions.`
# 
# `There are two ways to create a Sequential model. Let's take a look at each of them.`

# ![title](img/kerasmodelling.png)

# ### Using Sequential model
# 
# `Firstly, you can pass a list of layer instances to the constructor, as in the following example.For now, we will just explain enough to allow you to understand what is happening here.`
# 
# `Acquire the data. MNIST is a dataset of hand-drawn numerals, each on a 28 x 28 pixel grid. Every individual data point is an unsigned 8-bit integer (uint8), as are the labels:`

# #### Loading the datset

# In[7]:


mnist_data = tf.keras.datasets.mnist
(train_x,train_y), (test_x, test_y) = mnist_data.load_data()


# #### Definning the variables

# In[53]:


epochs=10
batch_size = 32


# In[8]:


# normalize all the data points and cast the labels to int64
train_x, test_x = tf.cast(train_x/255.0, tf.float32), tf.cast(test_x/255.0, tf.float32)
train_y, test_y = tf.cast(train_y,tf.int64),tf.cast(test_y,tf.int64)


# #### Building the Architecture 

# In[62]:


mnistmodel1 = tf.keras.models.Sequential([
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512,activation=tf.nn.relu),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])


# #### Compiling the model

# In[63]:


optimiser = tf.keras.optimizers.Adam()
mnistmodel1.compile (optimizer= optimiser, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])


# #### Fitting the model

# In[58]:


mnistmodel1.fit(train_x, train_y, batch_size=32, epochs=5)


# #### Evaluate the mnistmodel1

# In[64]:


mnistmodel1.evaluate(test_x, test_y)


# >This represents a loss of 0.09 and an accuracy of 0.9801 on the test data. 
# 
# >An accuracy of 0.98 means that out of 100 test data points, 98 were, on average, correctly identified by the model.

# In[ ]:


# The second way to create a Sequential model
# The alternative to passing a list of layers to the Sequential model's constructor is to use the add method, as follows, for the same architecture:


# #### Building the Architecture & Compiling

# In[51]:


mnistmodel2 = tf.keras.models.Sequential();
mnistmodel2.add(tf.keras.layers.Flatten())
mnistmodel2.add(tf.keras.layers.Dense(512, activation='relu'))
mnistmodel2.add(tf.keras.layers.Dropout(0.2))
mnistmodel2.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
mnistmodel2.compile (optimizer= tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy',metrics = ['accuracy'])


# #### Fitting the mnistmodel2

# In[52]:


mnistmodel2.fit(train_x, train_y, batch_size=64, epochs=5)


# #### Evaluate the mnistmodel2

# In[53]:


mnistmodel2.evaluate(test_x, test_y)


# ![title](img/kerasmodelling2.png)

# ### Keras functional API
# 
# `The functional API lets you build much more complex architectures than the simple linear stack of Sequential models we have seen previously. It also supports more advanced models. These models include multi-input and multi-output models, models with shared layers, and models with residual connections.`

# In[ ]:


import tensorflow as tf
mnist = tf.keras.datasets.mnist
(train_x,train_y), (test_x, test_y) = mnist.load_data()
train_x, test_x = train_x/255.0, test_x/255.0
epochs=10


# #### Building the Architecture

# In[56]:


inputs = tf.keras.Input(shape=(28,28)) # Returns a 'placeholder' tensor
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(512, activation='relu',name='d1')(x)
x = tf.keras.layers.Dropout(0.2)(x)
predictions = tf.keras.layers.Dense(10,activation=tf.nn.softmax, name='d2')(x)
mnistmodel3 = tf.keras.Model(inputs=inputs, outputs=predictions)


# #### Compile & Fit

# In[ ]:


optimiser = tf.keras.optimizers.Adam()
mnistmodel3.compile (optimizer= optimiser, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
mnistmodel3.fit(train_x, train_y, batch_size=32, epochs=epochs)


# #### Evaluate the mnistmodel3

# In[ ]:


mnistmodel3.evaluate(test_x, test_y)


# ### Subclassing the Keras Model class


import tensorflow as tf


# #### Building the subclass architecture




class MNISTModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MNISTModel, self).__init__()
        # Define your layers here.
        inputs = tf.keras.Input(shape=(28,28)) # Returns a placeholder tensor
        self.x0 = tf.keras.layers.Flatten()
        self.x1 = tf.keras.layers.Dense(512, activation='relu',name='d1')
        self.x2 = tf.keras.layers.Dropout(0.2)
        self.predictions = tf.keras.layers.Dense(10,activation=tf.nn.softmax, name='d2')


    def call(self, inputs):
    # This is where to define your forward pass
    # using the layers previously defined in `__init__`
        x = self.x0(inputs)
        x = self.x1(x)
        x = self.x2(x)
        return self.predictions(x)


# In[69]:


mnistmodel4 = MNISTModel()


# #### Compile & Fit

# In[71]:


batch_size = 32
steps_per_epoch = len(train_x.numpy())//batch_size
print(steps_per_epoch)
mnistmodel4.compile (optimizer= tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy',metrics = ['accuracy'])
mnistmodel4.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)


# #### Evaluate the mnistmodel4

# In[ ]:


mnistmodel4.evaluate(test_x, test_y)

