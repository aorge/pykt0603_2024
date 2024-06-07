import numpy
import keras

# from keras.models import Sequential

FILE_PATH = "./data/diabetes.csv"
dataset1 = numpy.loadtxt(FILE_PATH, delimiter=',', skiprows=1)
print(dataset1.shape)
inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]
model = keras.models.Sequential()
model.add(keras.layers.Dense(12, input_dim=8, activation='relu'))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(inputList, resultList, epochs=200, batch_size=20, validation_split=0.1)

scores = model.evaluate(inputList, resultList)
print(scores)
print(type(model.metrics_names), model.metrics_names)
print("accuracy", scores[1] * 100)
print(model.metrics_names[0], scores[0])