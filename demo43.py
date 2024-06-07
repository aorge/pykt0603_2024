import tensorflow as tf
import numpy as np

l1 = [5, 3, 8]
l2 = [3, -1, 2]
a1 = np.array(l1)
a2 = np.array(l2)
a3 = a1 + a2
a4 = np.add(a1, a2)

t1 = tf.constant(l1)
t2 = tf.constant(l2)
t3 = tf.add(t1, t2)
print("a3", a3)
print("a4", a4)
print("tensor t3", t3)
print("tensor t3 value", t3.numpy())
t4 = t1 + t2
print("t4 numpy:", t4.numpy())
t5 = np.add(t1.numpy(), t2.numpy())
print("t5 numpy:", t5)