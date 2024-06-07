from keras import datasets, utils, Sequential, layers, callbacks
import tensorflow as tf
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

flattenDim = 28 * 28
TRAINING_SIZE = len(train_images)
TEST_SIZE = len(test_images)

trainImages = np.reshape(train_images, (TRAINING_SIZE, flattenDim))
testImages = np.reshape(test_images, (TEST_SIZE, flattenDim))
print(type(trainImages[0]))
trainImages = trainImages.astype(np.float32)
testImages = testImages.astype(np.float32)
trainImages /= 255
testImages /= 255
print(trainImages[0])
NUM_DIGIT = 10

trainLabels = utils.to_categorical(train_labels, NUM_DIGIT)
testLabels = utils.to_categorical(test_labels, NUM_DIGIT)
model = Sequential()
model.add(layers.Dense(units=128, activation=tf.nn.relu, input_shape=(flattenDim,)))
model.add(layers.Dense(units=10, activation=tf.nn.softmax))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

tbCallback = callbacks.TensorBoard(log_dir="logs", histogram_freq=0, write_graph=True, write_images=True)
model.fit(trainImages, trainLabels, validation_split=0.1, epochs=10, batch_size=64, callbacks=[tbCallback])