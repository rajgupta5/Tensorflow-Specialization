#Extract Tranform Load

import tensorflow as tf
print(tf.__version__)
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt



dataset = tfds.load(name='mnist', split='train')
print(dataset)

assert isinstance(dataset, tf.data.Dataset)

print(tfds.list_builders())

mnist,info = tfds.load(name='mnist', split='train', with_info=True)
print(info)

print(info.homepage)
print(info.features['image'])
print(info.features['label'])
print(info.splits['train'].num_examples)
print(info.splits['test'].num_examples)


# as supervised returns dataset in (image,;abel) tuple otherwise will return a dictionary
mnist1 = tfds.load(name='mnist', as_supervised=True)
for image,label in mnist1['train'].take(2):
    print(image.shape, label.shape)


split = tfds.Split('test')
mnist2 = tfds.load(name='mnist', split=split)
print(mnist2)


# dataset builder
mnist_builder = tfds.builder('mnist')
mnist_builder.download_and_prepare()
mnist4 = mnist_builder.as_dataset(split=tfds.Split.TRAIN)
print(mnist4)

# EXTRACT
dataset = tfds.load(name="mnist", split="train")
# TRANSFORM
dataset.shuffle(100)
# LOAD
for data in dataset.take(1):
    image = data["image"].numpy().squeeze()
    label = data["label"].numpy()

    print("Label: {}".format(label))
    plt.imshow(image, cmap=plt.cm.binary)
    # plt.show()