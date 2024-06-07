import tensorflow as tf
import numpy as np


@tf.function
def add(p, q):
    return tf.math.add(p, q)


l1 = [1, 2, 3]
l2 = [4, 5, 6]
print(add(l1, l2))
print(add(np.array(l1), np.array(l2)))
print(add(tf.constant(l1), tf.constant(l2)))