from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, AveragePooling2D, Flatten

inputs = Input(shape=(16, 16, 3), name='input')
h = Conv2D(32, 3, activation='relu', name='con2d')(inputs)
h = AveragePooling2D(3, name='avgpool')(h)
outputs = Flatten(name='flatten')(h)
model = Model(inputs=inputs, outputs=outputs)

# print(model.layers[1].weights)
# print(model.layers[1].get_weights())
# print(model.layers[1].kernel)
# print(model.layers[1].bias)

# print(model.layers)
# print(model.layers[3].weights)

# print(model.get_layer('con2d').weights)

# print(model.get_layer('con2d').input.shape)
# print(model.get_layer('con2d').output.shape)

# creating new model using output of previous model
flatten_output = model.get_layer('flatten').output
model2 = Model(inputs=inputs, outputs=flatten_output)

# creating new model using prev model
model3 = Sequential([
    model2,
    Dense(10, activation='softmax', name='new_dense_layer')
])

#creating new model using prev model and functional api
new_outputs = Dense(10, activation='softmax', name='new_dense_layer')(model2.output)
model4 = Model(inputs=model2.input, outputs=new_outputs)


new_outputs = Dense(10, activation='softmax', name='new_dense_layer')(flatten_output)
model5 = Model(inputs=model.input, outputs=new_outputs)


