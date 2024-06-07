import tensorflow as tf




c1 = tf.constant("Hello Tensorflow")
c2 = tf.constant(100)
c3 = tf.constant(3.14159)
print(c1)
print(c2)
print(c3)
print(c1.numpy())
print(c2.numpy())
print(c3.numpy())