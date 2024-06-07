import tensorflow as tf

x = tf.Variable(0.)
print(type(x), x, x.numpy())

with tf.GradientTape() as tape:
    y = 2 * x + 3
    diff_x = tape.gradient(y, x)
    print("diff_x result={}".format(diff_x.numpy()))

x2 = tf.Variable(tf.random.uniform((2, 2)))
print(x2)
with tf.GradientTape() as tape:
    y2 = 5 * (x2) ** 2 + 4
    diff2 = tape.gradient(y2, x2)
print(diff2)