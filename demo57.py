import numpy
import keras
from sklearn.model_selection import train_test_split

# from keras.models import Sequential

FILE_PATH = "./data/diabetes.csv"
dataset1 = numpy.loadtxt(FILE_PATH, delimiter=',', skiprows=1)
print(dataset1.shape)
inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]
feature_train, feature_test, label_train, label_test = train_test_split(inputList, resultList, test_size=0.33,
                                                                        stratify=resultList)

for d in [resultList, label_train, label_test]:
    classes, counts = numpy.unique(d, return_counts=True)
    for cl, co in zip(classes, counts):
        print(f"{int(cl)}==>{co / sum(counts)}")

model = keras.models.Sequential()
model.add(keras.layers.Dense(12, input_dim=8, activation='relu'))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(feature_train, label_train, epochs=200, batch_size=20, validation_data=(feature_test, label_test))

scores = model.evaluate(feature_test, label_test)
print(scores)
print(type(model.metrics_names), model.metrics_names)
print(model.metrics_names[1], scores[1] * 100)
print(model.metrics_names[0], scores[0])