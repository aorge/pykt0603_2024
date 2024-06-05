from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from matplotlib import pyplot as plt

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])
plt.plot(x, y)
plt.scatter(x, y)
regession1 = LinearRegression()
regession1.fit(x, y)
print(regession1.coef_, regession1.intercept_)
x_seq = np.array(np.arange(5, 55, 0.1)).reshape(-1, 1)
plt.plot(x, regession1.coef_ * x + regession1.intercept_)
score1 = regession1.score(x, y)
plt.title("score={}".format(score1))
plt.show()

t = PolynomialFeatures(degree=2, include_bias=False)
t.fit(x)
x_ = t.transform(x)
print(f"x shape={x.shape}, x_ shape={x_.shape}")
print(x)
print(x_)
regression2 = LinearRegression()
regression2.fit(x_, y)
score2 = regression2.score(x_, y)
print(score2)

x_seq_ = t.transform(x_seq)
y_pred = regression2.predict(x_seq_)
plt.plot(x_seq, y_pred)
plt.plot(x, y)
plt.scatter(x, y)
plt.title("score={}".format(score2))
plt.show()