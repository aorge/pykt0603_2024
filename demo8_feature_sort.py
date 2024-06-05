from sklearn import datasets

d = datasets.make_regression(10, 6, noise=5)

X = d[0]
Y = d[1]
print(type(X), X.shape)

X_BY_X0 = sorted(X, key=lambda x: x[0])
X_BY_X2 = sorted(X, key=lambda x: x[2])
X_BY_X4 = sorted(X, key=lambda x: x[4])
print("xxxxxxx")
