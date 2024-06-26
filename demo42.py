import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()
l1 = [5, 3, 8]
l2 = [3, -1, 2]
a1 = np.array(l1)
a2 = np.array(l2)
a3 = a1 + a2
a4 = np.add(a1, a2)

print(a3)
print(a4)

t1 = tf.constant(l1)
t2 = tf.constant(l2)
t3 = tf.add(t1, t2)
print(t3)
with tf.compat.v1.Session() as s1:
    print(s1.run(t3))