import tensorflow as tf
import tensorflow_datasets as tfds

print("\u2022 Using TensorFlow Version:", tf.__version__)

# mnist_builder = tfds.builder("mnist:3.*.*")
# print(mnist_builder.version.implements(tfds.core.Experiment.S3))


train_ds, test_ds = tfds.load('mnist:3.*.*', split=['train', 'test'])
print(len(list(train_ds)))
print(len(list(test_ds)))


combined = tfds.load('mnist:3.*.*', split='train+test')
print(len(list(combined)))


first10k = tfds.load('mnist:3.*.*', split='train[:10000]')
print(len(list(first10k)))


first20p = tfds.load('mnist:3.*.*', split='train[:20%]')
print(len(list(first20p)))


val_ds = tfds.load('mnist:3.*.*', split=['train[{}%:{}%]'.format(k, k+20) for k in range(0, 100, 20)])
train_ds = tfds.load('mnist:3.*.*', split=['train[:{}%]+train[{}%:]'.format(k, k+20) for k in range(0, 100, 20)])
print(len(list(val_ds)))
print(len(list(train_ds)))


composed_ds = tfds.load('mnist:3.*.*', split='test[:10%]+train[-80%:]')
print(len(list(composed_ds)))