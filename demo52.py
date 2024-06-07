import tensorflow as tf

t = tf.Variable(5.)

with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        p = 4.9 * (t ** 2)
        speed = inner_tape.gradient(p, t)
        print("speed type={}".format(type(speed)))
        print("speed={}".format(speed.numpy()))
    a = outer_tape.gradient(speed, t)
    print("accelerate={}".format(a.numpy()))