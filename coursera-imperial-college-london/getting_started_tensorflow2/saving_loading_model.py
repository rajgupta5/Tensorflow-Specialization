import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.python.keras.models import load_model
import numpy as np

print(tf.__version__)

# Import the CIFAR-10 dataset and rescale the pixel values
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# # Use smaller subset -- speeds things up
# x_train = x_train[:10000]
# y_train = y_train[:10000]
# x_test = x_test[:1000]
# y_test = y_test[:1000]
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# # Plot the first 10 CIFAR-10 images
# fig, ax = plt.subplots(1, 10, figsize=(10, 1))
# for i in range(10):
#     ax[i].set_axis_off()
#     ax[i].imshow(x_train[i])
# plt.show()



# Introduce function to test model accuracy
def get_test_accuracy(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x=x_test, y=y_test, verbose=0)
    print('accuracy: {acc:0.3f}'.format(acc=test_acc))


# Introduce function that creates a new instance of a simple CNN
def get_new_model():
    model = Sequential([
        Conv2D(filters=16, input_shape=(32, 32, 3), kernel_size=(3, 3),
               activation='relu', name='conv_1'),
        Conv2D(filters=8, kernel_size=(3, 3), activation='relu', name='conv_2'),
        MaxPooling2D(pool_size=(4, 4), name='pool_1'),
        Flatten(name='flatten'),
        Dense(units=32, activation='relu', name='dense_1'),
        Dense(units=10, activation='softmax', name='dense_2')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Create an instance of the model and show model summary
# model = get_new_model()
# print(model.summary())
#
# # Test accuracy of the untrained model, around 10% (random)
# print(get_test_accuracy(model, x_test, y_test))

#
# # Create Tensorflow checkpoint object
# checkpoint_path = "model_checkpoint/checkpoint"
# checkpoint = ModelCheckpoint(filepath = checkpoint_path, frequency="epoch", save_weights_only=True, verbose=1)
#
# # Fit model, with simple checkpoint which saves (and overwrites) model weights every epoch
# history = model.fit(x_train, y_train, epochs = 3, verbose=2, callbacks=[checkpoint])
#
# # # Have a look at what the checkpoint creates
# # !ls -ltr model_checkpoint
#
# # Evaluate the performance of the trained model
# print(get_test_accuracy(model, x_test,y_test))
#
# # Create a new instance of the (initialised) model, accuracy around 10% again
# model = get_new_model()
# print(get_test_accuracy(model, x_test, y_test))
#
# # Load weights -- accuracy is the same as the trained model
# model.load_weights(checkpoint_path)
# print(get_test_accuracy(model, x_test, y_test))
#
# # !rm -r model_checkpoint

# model = get_new_model()
# checkpoint_5000_path = 'model_checkpoints_5000/checkpoint_{epoch:02d}_{batch:04d}'
# checkpoint_5000 = ModelCheckpoint(filepath=checkpoint_5000_path, save_weights_only=True, save_freq=5000, verbose=1)
# model.fit(x=x_train, y=y_train, epochs=3, validation_data=(x_test, y_test), batch_size=10, callbacks=[checkpoint_5000])


x_train = x_train[:100]
y_train = y_train[:100]
x_test = x_test[:100]
y_test = y_test[:100]

# model = get_new_model()
# checkpoint_best_path = 'model_checkpoint_best/checkpoint'
# checkpoint_best = ModelCheckpoint(filepath=checkpoint_best_path, save_weights_only=True, save_freq='epoch', \
#                                   monitor='val_accuracy', save_best_only=True, verbose=1)
# history = model.fit(x=x_train, y=y_train, epochs=50, validation_data=(x_test, y_test), batch_size=10, callbacks=[checkpoint_best], verbose=0)
#
# df = pd.DataFrame(history.history)
# df.plot(y=['accuracy', 'val_accuracy'])
# plt.show()
#
# new_model = get_new_model()
# new_model.load_weights(checkpoint_best_path)
# print(get_test_accuracy(new_model, x_test, y_test))

# model = get_new_model()
# checkpoint_path = 'model_checkpoints'
# checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False, frequency='epoch', \
#                                      verbose=1)
#
# model.fit(x=x_train, y=y_train, epochs=3, callbacks=[checkpoint], verbose=0)
# print(get_test_accuracy(model, x_test, y_test))
#
# new_model = load_model(checkpoint_path)
# print(get_test_accuracy(new_model, x_test, y_test))
#
# model.save('my_model.h5')
# new_model1 = load_model('my_model.h5')
# print(get_test_accuracy(new_model1, x_test, y_test))

model = ResNet50(weights='imagenet', include_top=True)
from tensorflow.keras.preprocessing import image
img_input = image.load_img('my_picture.jpg', target_size=(224,224))
img_input = image.img_to_array(img_input)
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
preprocess_input(img_input[np.newaxis,...])
preds = model.predict(img_input)
decode_predictions = decode_predictions(preds, top=3)[0]



from tensorflow.keras.applications import ResNet50
model = ResNet50(weights='imagenet')


from tensorflow.keras.preprocessing.image import load_img

lemon_img = load_img('lemon.jpg', target_size=(224, 224))
viaduct_img = load_img('viaduct.jpg', target_size=(224, 224))
water_tower_img = load_img('water_tower.jpg', target_size=(224, 224))

# Useful function: presents top 5 predictions and probabilities

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import pandas as pd

def get_top_5_predictions(img):
    x = img_to_array(img)[np.newaxis, ...]
    x = preprocess_input(x)
    preds = decode_predictions(model.predict(x), top=5)
    top_preds = pd.DataFrame(columns=['prediction', 'probability'],
                             index=np.arange(5)+1)
    for i in range(5):
        top_preds.loc[i+1, 'prediction'] = preds[0][i][1]
        top_preds.loc[i+1, 'probability'] = preds[0][i][2]
    return top_preds

model.get_layer('stateful_rnn').reset_states()

