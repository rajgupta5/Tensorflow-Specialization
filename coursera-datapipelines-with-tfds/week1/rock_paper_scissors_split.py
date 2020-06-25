import tensorflow as tf
import tensorflow_datasets as tfds

data, info = tfds.load(name='rock_paper_scissors:3.*.*', split='train', with_info=True)
print(info)

for x in info.splits:
  print(x + ":"+ str(info.splits[x].num_examples))

# rps_builder = tfds.builder('rock_paper_scissors:3.*.*')
# print(rps_builder.version.implements(tfds.core.Experiment.S3))

small_train = tfds.load('rock_paper_scissors:3.*.*', split='train[:10%]')
small_test = tfds.load('rock_paper_scissors:3.*.*', split='test[:10%]')
print(len(list(small_train)))
print(len(list(small_test)))


new_train = tfds.load('rock_paper_scissors:3.*.*', split='train[:90%]')
new_test = tfds.load('rock_paper_scissors:3.*.*', split='test[:90%]')
validation = tfds.load('rock_paper_scissors:3.*.*', split='train[-10%:] + test[-10%:]')
print(len(list(new_train)))
print(len(list(new_test)))
print(len(list(validation)))
