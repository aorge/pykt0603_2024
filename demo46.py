import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def computeArea(sides):
    a = sides[:, 0]
    b = sides[:, 1]
    c = sides[:, 2]
    s = (a + b + c) / 2
    areaSquare = s * (s - a) * (s - b) * (s - c)
    return areaSquare ** 0.5


with tf.compat.v1.Session() as session1:
    area = computeArea(tf.compat.v1.constant([
        [3.0, 4.0, 5.0],
        [6.0, 6.0, 6.0],
        [2.3, 4.1, 4.8],
        [4.0, 9.0, 9.0]
    ]))
    print(type(area), area)
    result = session1.run(area)
    print(result)