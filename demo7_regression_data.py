from sklearn import linear_model, datasets
from matplotlib import pyplot as plt
import numpy

regressionData1 = datasets.make_regression(100, 1, noise=20)
print(type(regressionData1), len(regressionData1))
print(type(regressionData1[0]), type(regressionData1[1]))
print(regressionData1[0].shape, regressionData1[1].shape)
plt.scatter(regressionData1[0], regressionData1[1], c='red', marker='^')

regression1 = linear_model.LinearRegression()
regression1.fit(regressionData1[0], regressionData1[1])
print("coef={}".format(regression1.coef_))
print("intercept={}".format(regression1.intercept_))
range1 = numpy.arange(regressionData1[0].min() - 0.5, regressionData1[0].max() + 0.5, 0.01)
plt.plot(range1, regression1.coef_ * range1 + regression1.intercept_)
score = regression1.score(regressionData1[0], regressionData1[1])
plt.title("score={}".format(score))
plt.show()