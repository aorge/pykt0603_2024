import pandas as pd
import keras
from sklearn.preprocessing import LabelBinarizer

from keras import callbacks, Sequential, layers

csv = pd.read_csv("data/bmi.csv")
csv['height'] = csv['height'] / 200
csv['weight'] = csv['weight'] / 100

encoder = LabelBinarizer()
transformedLabel = encoder.fit_transform(csv['label'])
print(csv['label'][:10])
print(transformedLabel[:10])

CUT = 25000
test_csv = csv[CUT:]
test_pat = test_csv[['weight', 'height']]
test_ans = transformedLabel[CUT:]
train_csv = csv[:CUT]
train_pat = train_csv[['weight', 'height']]
train_ans = transformedLabel[:CUT]

model = Sequential()
model.add(layers.Dense(10, activation='relu', input_shape=(2,)))
model.add(layers.Dense(3, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
tensorboard = callbacks.TensorBoard(log_dir='logs')
model.fit(train_pat, train_ans, batch_size=50, epochs=100, verbose=1,
          validation_data=(test_pat, test_ans), callbacks=[tensorboard])