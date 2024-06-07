import tensorflow as tf

W = tf.Variable(tf.random.uniform((1, 1)))
b = tf.Variable(tf.zeros(1, ))
x = tf.random.uniform((1, 1))
print(W, b, x)
print('------------------------------------------------')
with tf.GradientTape() as tape:
    y = tf.matmul(x, W) + 2 * b
    grad_of_y_wrt_w_and_b = tape.gradient(y, [W, b])
print('------------------------------------------------')
print("x=", x.numpy()[0][0])
print('------------------------------------------------')
print("W=",W.numpy())
print('------------------------------------------------')
print("b=",b.numpy())
print('------------------------------------------------')
print("y=",y.numpy())
print('------------------------------------------------')
print("dy/dw=",grad_of_y_wrt_w_and_b[0].numpy())
print('------------------------------------------------')
print("dy/db=",grad_of_y_wrt_w_and_b[1].numpy())