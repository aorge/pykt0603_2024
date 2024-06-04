import os
import sklearn
import numpy
import tensorflow as tf
import keras
from sklearn.datasets import load_iris


print("工作目錄是:{}".format(format(os.getcwd())))
print("sklearn的版本是:{}".format(format(sklearn.__version__)))
print("numpy的版本是:{}".format(format(numpy.__version__)))
print("tensorflow的版本是:{}".format(format(tf.__version__)))
print("keras的版本是:{}".format(format(keras.__version__)))

iris = load_iris()
print(type(iris))

print(tf.reduce_mean(tf.random.normal([1000, 1000])).numpy())

