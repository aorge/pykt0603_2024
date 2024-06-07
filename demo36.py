from matplotlib import pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# Y = np.array([1, 1, 1, 2, 2, 2])
# Y = np.array([1, 2, 2, 1, 2, 2])
Y = np.array([1, 1, 2, 1, 1, 2])
x_min = -4
x_max = 4
y_min = -4
y_max = 4
h = .020
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
c1 = GaussianNB()
c1.fit(X, Y)
Z = c1.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.pcolormesh(xx, yy, Z)

XB = []
YB = []
XR = []
YR = []

index = 0
for index in range(0, len(Y)):
    if Y[index] == 1:
        XB.append(X[index, 0])
        YB.append(X[index, 1])
    if Y[index] == 2:
        XR.append(X[index, 0])
        YR.append(X[index, 1])

plt.scatter(XB, YB, color='b', label="label1:blue")
plt.scatter(XR, YR, color='r', label="label2:red")
plt.legend()

plt.show()