import numpy as np
import tensorflow as tf

scores = [3.0, 1.0, 2.0]


def manualSoftMax(x):
    x = np.array(x)
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def manualNoMax(x):
    x = np.array(x)
    return x / np.sum(x, axis=0)


print(manualNoMax(scores))
print(manualSoftMax(scores))
print(tf.nn.softmax(scores).numpy())