import tensorflow as tf
from datetime import datetime


@tf.function
def computeArea(sides):
    a = sides[:, 0]
    b = sides[:, 1]
    c = sides[:, 2]
    s = (a + b + c) / 2
    areaSquare = s * (s - a) * (s - b) * (s - c)
    return areaSquare ** 0.5


triangles = [
    [3.0, 4.0, 5.0],
    [6.0, 6.0, 6.0],
    [2.3, 4.1, 4.8],
    [4.0, 9.0, 9.0]
]

# 手動建立logs目錄
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = 'logs/demo47/%s' % stamp
writer = tf.summary.create_file_writer(logdir)

tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=logdir)
print(computeArea(tf.constant(triangles)).numpy())

with writer.as_default():
    tf.summary.trace_export(name="my_func", step=0)
    tf.summary.trace_off()