from matplotlib import pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

print(dir(iris))
labels = iris.feature_names
print('-------------------------------------')
print(labels)
X = iris.data
print('-------------------------------------')
print(len(X), X.shape)
species = iris.target
print('-------------------------------------')
print(iris.target_names)

counter = 1
for i in range(4):
    for j in range(i + 1, 4):
        xData = X[:, i]
        yData = X[:, j]
        x_min, x_max = xData.min() - 0.5, xData.max() + 0.5
        y_min, y_max = yData.min() - 0.5, yData.max() + 0.5
        plt.scatter(xData, yData, c=species, cmap=plt.cm.Paired)
        plt.xlabel(labels[i])
        plt.ylabel(labels[j])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()