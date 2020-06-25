import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG19
vgg_model = VGG19()


from tensorflow.keras.models import load_model,Model
# vgg_model=load_model('/Users/rajkgupta/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

vgg_input = vgg_model.input
vgg_layers = vgg_model.layers
print(vgg_model.summary())

layer_outputs = [layer.output for layer in vgg_layers]
features = Model(inputs=vgg_input, outputs=layer_outputs)

tf.keras.utils.plot_model(features, 'vgg19_model.png', show_shapes=True)

from PIL import Image
plt.imshow(Image.open('data/cool_cat.jpg'))
plt.show()

from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image

img_path = 'data/cool_cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

extracted_features = features(x)
f1 = extracted_features[0]
print('\n f1.shape: ', f1.shape)
imgs = f1[0, :,:]
plt.figure(figsize=(15,15))
for n in range(3):
    ax = plt.subplot(1,3,n+1)
    plt.imshow(imgs[:,:,n])
    plt.axis('off')
plt.subplots_adjust(wspace=0.01, hspace=0.01)
plt.show()

f2 = extracted_features[1]
print('\n f2.shape: ', f2.shape)
imgs = f2[0, :,:]
plt.figure(figsize=(15,15))
for n in range(16):
    ax = plt.subplot(4,4,n+1)
    plt.imshow(imgs[:,:,n])
    plt.axis('off')
plt.subplots_adjust(wspace=0.01, hspace=0.01)
plt.show()