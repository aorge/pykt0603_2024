from sklearn import datasets
from matplotlib import pyplot as plt

d = datasets.make_regression(10, 6, noise=5)

for i in range(6):
    x1 = d[0][:, i]
    y = d[1]
    plt.scatter(x1, y)
    plt.title("#{} feature".format(i))
    plt.show()
