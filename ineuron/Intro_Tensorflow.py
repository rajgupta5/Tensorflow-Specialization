import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.get_logger().setLevel('INFO')

print("TensorFlow version: {}".format(tf.__version__))
print("Keras version: {}".format(tf.keras.__version__))

variable = tf.Variable([3, 3])
# if tf.test.is_gpu_available():
#     print('GPU')
#     print('GPU #0?')
#     print(var.device.endswith('GPU:0'))
# else:
#     print('CPU')

ineuron = tf.constant(42)
print(ineuron)
print(ineuron.numpy())

ineuron1 = tf.constant(1, dtype=tf.int64)
print(ineuron1)
print(ineuron1.numpy())


ineuron_x = tf.constant([[4,2],[9,5]])
print(ineuron_x)
print(ineuron_x.numpy())