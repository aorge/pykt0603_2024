import numpy
import keras
from sklearn.model_selection import StratifiedKFold

# from keras.models import Sequential

FILE_PATH = "./data/diabetes.csv"
dataset1 = numpy.loadtxt(FILE_PATH, delimiter=',', skiprows=1)
print(dataset1.shape)
inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]

fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
totalScores = []

for train, test in fiveFold.split(inputList, resultList):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(12, input_dim=8, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(inputList[train], resultList[train], epochs=200, batch_size=20, verbose=0)

    scores = model.evaluate(inputList[test], resultList[test], verbose=0)
    totalScores.append(scores[1])
    print("accuracy={}".format(scores[1]))
print("total Scores={}".format(totalScores))
print("mean={}, std={}".format(numpy.mean(totalScores), numpy.std(totalScores)))