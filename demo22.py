from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt

iris = datasets.load_iris()

X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(int)
print(X)
print(y)
regression1 = LogisticRegression()
regression1.fit(X, y)
print(regression1.coef_, regression1.intercept_)
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_prob = regression1.predict_proba(X_new)

plt.plot(X, y, 'gs')
plt.plot(X_new, y_prob[:, 1], 'g--', label='iris-virginica')
plt.plot(X_new, y_prob[:, 0], 'b-', label='Not iris-virginica')
plt.xlabel("petal width")
plt.ylabel("probability")
plt.legend(loc="upper left")
plt.show()