import tensorflow as tf

ic = tf.constant(3.)
# ic = tf.Variable(3.)
with tf.GradientTape() as tape:
    tape.watch(ic)
    result = tf.square(ic)
    gradient = tape.gradient(result, ic)
print(gradient.numpy())