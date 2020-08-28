from tensorflow import keras
import pandas as pd
import numpy as np
import seaborn as sns


basepath = '/Users/rajkgupta/Downloads/data/examples/clusters/'

train_df = pd.read_csv(basepath + 'train.csv')
test_df = pd.read_csv(basepath + 'test.csv')

np.random.shuffle(train_df.values)

print(train_df.head())
print(train_df.color.unique())
sns.scatterplot(train_df.x, train_df.y, hue=train_df.color)

color_dict = {'red':0, 'blue':1, 'green':2, 'teal':3, 'orange':4, 'purple':5}
train_df.color = train_df.color.apply(lambda x: color_dict[x])
test_df.color = test_df.color.apply(lambda x: color_dict[x])

print(train_df.color.unique())
print(test_df.color.unique())

x_train = np.column_stack((train_df.x.values, train_df.y.values))
y_train = train_df['color'].values
x_test = np.column_stack((test_df.x.values, test_df.y.values))
y_test = test_df.color.values

model = keras.Sequential([
    keras.layers.Dense(32, input_shape=(2,), activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    # keras.layers.Dropout(0.1),
    keras.layers.Dense(6, activation='softmax')
])

model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train, epochs=20, verbose=2)

model.evaluate(x_test, y_test)

print(test_df.iloc[1])
print(pd.get_dummies(test_df.color).values[0])
print(np.round(model.predict(np.array([[-0.173868, 1.381852]]))))
