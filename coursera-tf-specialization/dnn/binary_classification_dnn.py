from tensorflow import keras
import pandas as pd
import numpy as np
import seaborn as sns


basepath = '/Users/rajkgupta/Downloads/data/examples/quadratic/'
train_df = pd.read_csv(basepath + 'train.csv')
test_df = pd.read_csv(basepath + 'test.csv')

np.random.shuffle(train_df.values)

print(train_df.head())
sns.scatterplot(train_df.x, train_df.y, hue=train_df.color)

x_train = np.column_stack((train_df.x.values, train_df.y.values))
y_train = train_df['color'].values
x_test = np.column_stack((test_df.x.values, test_df.y.values))
y_test = test_df.color.values

model = keras.Sequential([
    keras.layers.Dense(16, input_shape=(2,), activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    # keras.layers.Dropout(0.1),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train, epochs=20, verbose=2)

model.evaluate(x_test, y_test)

