from matplotlib import pyplot as plt
from sklearn import linear_model

regression1 = linear_model.LinearRegression()
print(type(regression1))
features = [[1], [2], [3], [8]]
values = [1, 4, 15, 40]
plt.scatter(features, values, c='green')
regression1.fit(features, values)
print(regression1.coef_)
print(regression1.intercept_)
r1 = [-1, 8]
plt.plot(r1, regression1.coef_ * r1 + regression1.intercept_, c='gray')
plt.show()