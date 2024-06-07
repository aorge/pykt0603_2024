import numpy
import keras

# from keras.models import Sequential

FILE_PATH = "./data/diabetes.csv"
dataset1 = numpy.loadtxt(FILE_PATH, delimiter=',', skiprows=1)
print(dataset1.shape)
inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]


def createModel():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(12, input_dim=8, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


m = createModel()
m.fit(inputList, resultList, epochs=200, batch_size=20)
# save model
# 手動建一個models目錄
MODEL_NAME = 'models/demo55.keras'
keras.models.save_model(m, MODEL_NAME)

scores = m.evaluate(inputList, resultList)
print(scores)
print(type(m.metrics_names), m.metrics_names)
print(m.metrics_names[1], scores[1] * 100)
print(m.metrics_names[0], scores[0])

m2 = createModel()
print("沒有訓練直接評估")
scores2 = m2.evaluate(inputList, resultList)
print(scores2)

print("載入現成的model評估")
m3 = keras.models.load_model(MODEL_NAME)
scores3 = m3.evaluate(inputList, resultList)
print(scores3)
keras.utils.plot_model(m3, to_file='output/demo55_m3.png', show_shapes=True, show_layer_names=True)