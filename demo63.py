import numpy
from keras import datasets
from matplotlib import pyplot

(X_train, y_train), (X_test, y_test) = datasets.imdb.load_data()
X = numpy.concatenate((X_train, X_test), axis=0)
y = numpy.concatenate((y_train, y_test), axis=0)
print(X[0])
print(X.shape)
print(y.shape)
print("不同的y", numpy.unique(y))
print(len(numpy.unique(numpy.hstack(X))))
result = [len(x) for x in X]
print("mean={}, std={}".format(numpy.mean(result), numpy.std(result)))

pyplot.subplot(121)
pyplot.boxplot(result)
pyplot.subplot(122)
pyplot.hist(result)
pyplot.show()