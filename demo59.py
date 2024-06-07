import numpy
import keras
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scikeras.wrappers import KerasClassifier

# from keras.models import Sequential

FILE_PATH = "./data/diabetes.csv"
dataset1 = numpy.loadtxt(FILE_PATH, delimiter=',', skiprows=1)
print(dataset1.shape)
inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]


def create_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(12, input_dim=8, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


m = KerasClassifier(model=create_model, epochs=200, batch_size=20, verbose=0)
print(type(m))
fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
results = cross_val_score(m, inputList, resultList, cv=fiveFold)

print(results)