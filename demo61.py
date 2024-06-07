from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras import utils
from keras import Sequential, layers
import keras
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score

DATA_FILE_PATH = "data/iris.data"
dataFrame = read_csv(DATA_FILE_PATH, header=None)
print(dataFrame.shape)
dataset = dataFrame.values
print(type(dataset))
print(dataFrame.head(10))
features = dataset[:, 0:4].astype(float)
labels = dataset[:, 4]
print(labels)

encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
print(encoded_Y)
print(type(encoded_Y), np.unique(encoded_Y, return_counts=True))
dummy_y = utils.to_categorical(encoded_Y)
print(dummy_y)


def baseline_model():
    model = Sequential()
    model.add(layers.Dense(8, input_dim=4, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    print(model.summary())
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    return model


e = KerasClassifier(model=baseline_model, epochs=400, batch_size=10, verbose=1)
kfold = KFold(n_splits=3, shuffle=True)
results = cross_val_score(e, features, dummy_y, cv=kfold)
print(results)
