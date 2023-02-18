import keras.layers
import tensorflow.python.training.optimizer
from keras.models import Sequential
from keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D
import keras
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import RMSprop


from sklearn.model_selection import train_test_split
import sys

filename = "newsudoku.csv"

chunksize = 10 ** 6
chunk = pd.read_csv(filename, chunksize=chunksize)
results = pd.concat(chunk)
X = results['puzzle']
X = X.to_numpy()
numbers = [[int(x) for x in s] for s in X]
X = np.array(numbers)
X = np.reshape(X, (X.shape[0], 9, 9, 1))
y = results['solution']
y = y.to_numpy()
numbers = [[int(x) for x in s] for s in y]
y = np.array(numbers)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


model = keras.models.Sequential([
    keras.layers.Input(shape=(9, 9,1,)),
    keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (1,1), activation="relu", padding="same"),
    keras.layers.Flatten(),
    keras.layers.Dense(9 * 81, activation="sigmoid"),
    keras.layers.Reshape((81, 9)),
    keras.layers.Activation("softmax"),
])

model.compile(optimizer=keras.optimizers.Adam(lr=0.01),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=512, validation_split=0.2)
model.evaluate(x_test, y_test)
pred = model.predict(np.array([x_test[0, :]]))
print("puzzle: \n",x_test[0, :].reshape(81,))
print("solution: \n",y_test[0, :])
solution = []
for element in pred:
    for child in element:
        solution.append(np.argmax(child))
solution = np.array(solution)
print("pred: \n", solution)
x_new_test = x_test[0, :].reshape(81,)
for item in range(x_new_test.shape[0]):
    if x_new_test.item(item) == 0:
        print(x_new_test.item(item), y_test[0, item], solution[item])