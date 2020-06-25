import tensorflow as tf
import tensorflow_datasets as tfds

print("\u2022 Using TensorFlow Version:", tf.__version__)

all_splits = tfds.Split.TRAIN + tfds.Split.TEST
ds = tfds.load("mnist", split=all_splits)
print("Number of Records: {:,}".format(len(list(ds))))


s1, s2, s3, s4 = tfds.Split.TRAIN.subsplit(k=4)
dataset_split_1 = tfds.load("mnist", split=s1)
dataset_split_2 = tfds.load("mnist", split=s2)
dataset_split_3 = tfds.load("mnist", split=s3)
dataset_split_4 = tfds.load("mnist", split=s4)
print(len(list(dataset_split_1)))
print(len(list(dataset_split_2)))
print(len(list(dataset_split_3)))
print(len(list(dataset_split_4)))


s1 = tfds.Split.TRAIN.subsplit(tfds.percent[0:25])
s2 = tfds.Split.TRAIN.subsplit(tfds.percent[25:50])
s3 = tfds.Split.TRAIN.subsplit(tfds.percent[50:75])
s4 = tfds.Split.TRAIN.subsplit(tfds.percent[75:100])
dataset_split_1 = tfds.load("mnist", split=s1)
dataset_split_2 = tfds.load("mnist", split=s2)
dataset_split_3 = tfds.load("mnist", split=s3)
dataset_split_4 = tfds.load("mnist", split=s4)
print(len(list(dataset_split_1)))
print(len(list(dataset_split_2)))
print(len(list(dataset_split_3)))
print(len(list(dataset_split_4)))