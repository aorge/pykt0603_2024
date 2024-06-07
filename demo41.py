import tensorflow as tf

tf.compat.v1.disable_eager_execution()
c1 = tf.constant("Hello Tensorflow")
c2 = tf.constant(100)
c3 = tf.constant(3.14159)
print(type(c1), c1)
print(type(c2), c2)
print(type(c3), c3)
s1 = tf.compat.v1.Session()
print(type(s1))
print(s1.run(c1))
print(s1.run(c2))
print(s1.run(c3))
s1.close()

with tf.compat.v1.Session() as s2:
    print(s2.run(c1))
    print(s2.run(c2))
    print(s2.run(c3))