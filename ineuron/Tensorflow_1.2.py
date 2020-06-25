#!/usr/bin/env python
# coding: utf-8

# ![title](img/tensorflow4.png)

# # Using Data pipelines

# > Data may also be passed into the fit method as a tf.data.Dataset() iterator
# > The from_tensor_slices() method converts the NumPy arrays into a dataset
# > The batch() and shuffle() methods chained together. 
# 
# >Next, the map() method invokes a method on the input images, x, that randomly flips one in two of them across
# the y-axis, effectively increasing the size of the image set
# 
# >Finally, the repeat() method means that the dataset will be re-fed from the beginning when its end is
# reached (continuously)

# In[1]:


import tensorflow as tf
mnist = tf.keras.datasets.mnist
(train_x,train_y), (test_x, test_y) = mnist.load_data()
train_x, test_x = train_x/255.0, test_x/255.0
epochs=10


# In[3]:


batch_size = 32
buffer_size = 10000
training_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(32).shuffle(10000)
training_dataset = training_dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
training_dataset = training_dataset.repeat()


# In[5]:


testing_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(batch_size).shuffle(10000)
testing_dataset = training_dataset.repeat()


# #### Building the model architecture

# In[6]:


#Now in the fit() function, we can pass the dataset directly in, as follows:
model5 = tf.keras.models.Sequential([
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(512,activation=tf.nn.relu),
 tf.keras.layers.Dropout(0.2),
 tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])


# #### Compiling the model

# In[7]:


steps_per_epoch = len(train_x)//batch_size #required becuase of the repeat() on the dataset
optimiser = tf.keras.optimizers.Adam()
model5.compile (optimizer= optimiser, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])


# #### Fitting the model

# In[8]:


model5.fit(training_dataset, epochs=epochs, steps_per_epoch = steps_per_epoch)


# #### Evaluating the model

# In[9]:


model5.evaluate(testing_dataset,steps=10)


# In[10]:


import datetime as dt
callbacks = [
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='log/{}/'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
]


# In[11]:


model5.fit(training_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch,
          validation_data=testing_dataset,
          validation_steps=3)


# #### Evaluating

# In[12]:


model5.evaluate(testing_dataset,steps=10)


# ## Saving and loading Keras models

# >The Keras API in TensorFlow has the ability to save and restore models easily. This is done as follows, and saves the model in the current directory. Of course, a longer path may be passed here:
# 
# #### Saving a model
#     
# `model.save('./model_name.h5')`
# 
# >This will save the model architecture, its weights, its training state (loss, optimizer), and the state of the optimizer, so that you can carry on training the model from where you left off.
# 

# >Loading a saved model is done as follows. Note that if you have compiled your model, the load will compile your model using the saved training configuration:
# 
# #### Loding a model
# 
# `from tensorflow.keras.models import load_model
# new_model = load_model('./model_name.h5')`

# >It is also possible to save just the model weights and load them with this (in which case, you must build your architecture to load the weights into):
# 
# #### Saving the model weights only
#     
#     `model.save_weights('./model_weights.h5')`
#     
# >Then use the following to load it:
# 
# #### Loding the weights
#     
#     `model.load_weights('./model_weights.h5')`

# # Keras datasets
# 
# >The following datasets are available from within Keras: boston_housing, cifar10, cifar100, fashion_mnist, imdb, mnist,and reuters.
# 
# >They are all accessed with the function.
# 
# `load_data()`  
# 
# >For example, to load the fashion_mnist dataset, use the following:
# 
# `(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()`

# ![title](img/dataset.png)

# ### Using NumPy arrays with datasets

# In[13]:


import tensorflow as tf
import numpy as np
number_items = 11
number_list1 = np.arange(number_items)
number_list2 = np.arange(number_items,number_items*2)


# #### Create datasets, using the from_tensor_slices() method

# In[14]:


number_list1_dataset = tf.data.Dataset.from_tensor_slices(number_list1)


# #### Create an iterator on it using the make_one_shot_iterator() method:

# In[15]:


iterator = tf.compat.v1.data.make_one_shot_iterator(number_list1_dataset)


# #### Using them together, with the get_next method:

# In[16]:


for item in number_list1_dataset:
    number = iterator.get_next().numpy()
    print(number)


# >Note that executing this code twice in the same program run will raise an error because we are using a one-shot iterator

# #### It's also possible to access the data in batches() with the batch method. Note that the first argument is the number of elements to put in each batch and the second is the self-explanatory drop_remainder argument:

# In[18]:


number_list1_dataset = tf.data.Dataset.from_tensor_slices(number_list1).batch(3, drop_remainder = False)
iterator = tf.compat.v1.data.make_one_shot_iterator(number_list1_dataset)
for item in number_list1_dataset:
    number = iterator.get_next().numpy()
    print(number)


# ### There is also a zip method, which is useful for presenting features and labels together:

# In[19]:


data_set1 = [1,2,3,4,5]
data_set2 = ['a','e','i','o','u']
data_set1 = tf.data.Dataset.from_tensor_slices(data_set1)
data_set2 = tf.data.Dataset.from_tensor_slices(data_set2)
zipped_datasets = tf.data.Dataset.zip((data_set1, data_set2))
iterator = tf.compat.v1.data.make_one_shot_iterator(zipped_datasets)
for item in zipped_datasets:
    number = iterator.get_next()
    print(number)


# #### We can concatenate two datasets as follows, using the concatenate method:

# In[21]:


datas1 = tf.data.Dataset.from_tensor_slices([1,2,3,5,7,11,13,17])
datas2 = tf.data.Dataset.from_tensor_slices([19,23,29,31,37,41])
datas3 = datas1.concatenate(datas2)
print(datas3)
iterator = tf.compat.v1.data.make_one_shot_iterator(datas3)
for i in range(14):
    number = iterator.get_next()
    print(number)


# #### We can also do away with iterators altogether, as shown here:

# In[22]:


epochs=2
for e in range(epochs):
    for item in datas3:
        print(item)


# ### Using comma-separated value (CSV)files with datasets.
# 
# >CSV files are a very popular method of storing data. TensorFlow 2 contains flexible methods for dealing with them. 
# 
# >The main method here is tf.data.experimental.CsvDataset.

# #### CSV Example 1

# >With the following arguments, our dataset will consist of two items taken from each row of the
# filename file, both of the float type, with the first line of the file ignored and columns 1 and 2 used
# (column numbering is, of course, 0-based):
# 

# In[24]:


filename = ["./size_1000.csv"]
record_defaults = [tf.float32] * 2 # two required float columns
data_set = tf.data.experimental.CsvDataset(filename, record_defaults, header=True, select_cols=[1,2])
for item in data_set:
    print(item)


# #### #CSV example 2

# In[ ]:



# In this example, and with the following arguments, our dataset will consist of one required float,
# one optional float with a default value of 0.0, and an int, where there is no header in the CSV file and
# only columns 1, 2, and 3 are imported:
#file Chapter_2.ipynb


# In[22]:


filename = "mycsvfile.txt"
record_defaults = [tf.float32, tf.constant([0.0], dtype=tf.float32), tf.int32,]
data_set = tf.data.experimental.CsvDataset(filename, record_defaults, header=False, select_cols=[1,2,3])
for item in data_set:
    print(item)


# #### #CSV example 3

# In[24]:


#For our final example, our dataset will consist of two required floats and a required string, where the
#CSV file has a header variable:
filename = "file1.txt"
record_defaults = [tf.float32, tf.float32, tf.string ,]
dataset = tf.data.experimental.CsvDataset(filename, record_defaults, header=False)
for item in dataset:
    print(item[0].numpy(), item[1].numpy(),item[2].numpy().decode() )


# ## TFRecords

# >TFRecord format is a binary file format. For large files, it is a good choice because binary files take up less disc space, take less time to copy, and can be read very efficiently from the disc. All this can have a significant effect on the efficiency of your data pipeline and thus, the training time of your model. The format is also optimized in a
# variety of ways for use with TensorFlow. It is a little complex because data has to be converted into
# the binary format prior to storage and decoded when read back.

# #### #TFRecord example 1

# 
# >A TFRecord file is a sequence of binary strings, its structure must be specified prior to
# saving so that it can be properly written and subsequently read back.
# 
# >TensorFlow has two structures for this, 
# 
# `tf.train.Example and tf.train.SequenceExample. `
# 
# >We have to store each sample of your data in one of these structures, then serialize it, and use `tf.python_io.TFRecordWriter` to save it to disk.
# 
# >In the next example, 
# the  data, is first converted to the binary format and then saved to disc.
# 
# >A feature is a dictionary containing the data that is passed to tf.train.Example prior to serialization and saving the data.

# In[25]:


import tensorflow as tf
import numpy as np
data = np.array([10.,11.,12.,13.,14.,15.])
def npy_to_tfrecords(fname,data):
    writer = tf.io.TFRecordWriter(fname)
    feature={}
    feature['data'] = tf.train.Feature(float_list=tf.train.FloatList(value=data))
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized = example.SerializeToString()
    writer.write(serialized)
    writer.close()
npy_to_tfrecords("./myfile.tfrecords",data)


# >The code to read the record back is as follows. 
# 
# >A parse_function function is constructed that decodes the dataset read back from the file. This requires a dictionary (keys_to_features) with the same name and structure as the saved data:

# In[26]:


data_set = tf.data.TFRecordDataset("./myfile.tfrecords")
def parse_function(example_proto):
    keys_to_features = {'data':tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing = True) }
    parsed_features = tf.io.parse_single_example(serialized=example_proto, features=keys_to_features)
    return parsed_features['data']
data_set = data_set.map(parse_function)
iterator = tf.compat.v1.data.make_one_shot_iterator(data_set)
# array is retrieved as one item
item = iterator.get_next()
print(item)
print(item.numpy())
print(item[2].numpy())


# #### TFRecord example 2

# In[25]:


filename = './students.tfrecords'
dataset = {
'ID': 61553,
'Name': ['Jones', 'Felicity'],
'Scores': [45.6, 97.2]}


# >Using this, we can construct a tf.train.Example class, again using the `Feature()` method. Note how we have to encode our string:
# 

# In[27]:


ID = tf.train.Feature(int64_list=tf.train.Int64List(value=[dataset['ID']]))
Name = tf.train.Feature(bytes_list=tf.train.BytesList(value=[n.encode('utf-8') for n in dataset['Name']]))
Scores = tf.train.Feature(float_list=tf.train.FloatList(value=dataset['Scores']))
example = tf.train.Example(features=tf.train.Features(feature={'ID': ID, 'Name': Name, 'Scores': Scores }))


# #### #Serializing and writing this record to disc is the same as TFRecord example 1:

# In[29]:


writer_rec = tf.io.TFRecordWriter(filename)
writer_rec.write(example.SerializeToString())
writer_rec.close()


# #### To read this back, we just need to construct our parse_function function to reflect the structure of the record:

# In[38]:


data_set = tf.data.TFRecordDataset("./students.tfrecords")
def parse_function(example_proto):
    keys_to_features = {'ID':tf.io.FixedLenFeature([], dtype = tf.int64),
    'Name':tf.io.VarLenFeature(dtype = tf.string),
    'Scores':tf.io.VarLenFeature(dtype = tf.float32)}
    parsed_features = tf.io.parse_single_example(serialized=example_proto, features=keys_to_features)
    return parsed_features["ID"], parsed_features["Name"],parsed_features["Scores"]


# #### Parsing the data

# In[40]:


dataset = data_set.map(parse_function)
iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
items = iterator.get_next()


# ### Record is retrieved as one item

# In[41]:


print(items)


# >Now we can extract our data from item (note that the string must be decoded (from bytes) where the default for our Python 3 is utf8). Note also that the string and
# the array of floats are returned as sparse arrays, and to extract them from the record, we use the sparse array value method:

# In[42]:


print("ID: ",item[0].numpy())
name = item[1].values.numpy()
name1= name[0].decode()
name2 = name[1].decode('utf8')
print("Name:",name1,",",name2)
print("Scores: ",item[2].values.numpy())


# ### One-hot Encoding

# >One-hot encoding (OHE) is where a tensor is constructed from the data labels with a 1 in each of
# the elements corresponding to a label's value, and 0 everywhere else; that is, one of the bits in the
# tensor is hot (1).

# #### One-hot Encoding Example 1

# >In this example, we are converting a decimal value of 7 to a one-hot encoded value of 0000000100 using
# 
# `the tf.one_hot() method:`

# In[43]:


z = 7
z_train_ohe = tf.one_hot(z, depth=10).numpy()
print(z, "is ",z_train_ohe,"when one-hot encoded with a depth of 10")


# #### One-hot Encoding Example 2

# >Using the fashion MNIST dataset.
# 
# >The original labels are integers from 0 to 9, so, for example, a label of 5 becomes 0000010000 when onehot encoded, but note the difference between the index and the label stored at that index:

# In[66]:


import tensorflow as tf
from tensorflow.python.keras.datasets import fashion_mnist

width, height, = 28,28
# total classes
n_classes = 10


# #### loading the dataset

# In[ ]:


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# #### Split feature training set into training and validation sets

# In[ ]:


split = 50000
(y_train, y_valid) = y_train[:split], y_train[split:]


# #### one-hot encode the labels using TensorFlow then convert back to numpy for display

# In[68]:


y_train_ohe = tf.one_hot(y_train, depth=n_classes).numpy()
y_valid_ohe = tf.one_hot(y_valid, depth=n_classes).numpy()
y_test_ohe = tf.one_hot(y_test, depth=n_classes).numpy()

# show difference between the original label and a one-hot-encoded label
i=8
print(y_train[i]) # 'ordinary' number value of label at index i=8 is 5
# note the difference between the index of 8 and the label at that index which is 5
print(y_train_ohe[i]) 


# In[ ]:




