import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
c1 = GaussianNB()
c1.fit(X, Y)
newX = [[-5, -5], [-5, 5], [5, 5], [5, -5]]
print(c1.predict(newX))
print(c1.predict([[-0.5, -0.5]]))
c2 = GaussianNB()
c2.partial_fit(X, Y, np.unique(Y))
print(c2.predict([[-0.5, -0.5]]))
c2.partial_fit([[-0.75, -0.75]], [2])
print(c2.predict([[-0.5, -0.5]]))
