import numpy as np
from sklearn import linear_model, datasets
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from matplotlib import pyplot as plt

diabetes = datasets.load_diabetes()

dataForTest = -50

data_train = diabetes.data[:dataForTest]
target_train = diabetes.target[:dataForTest]

data_test = diabetes.data[dataForTest:]
target_test = diabetes.target[dataForTest:]

regression1 = linear_model.LinearRegression()
regression1.fit(data_train, target_train)
print(regression1.coef_)
print(regression1.intercept_)

for i, v in enumerate(regression1.coef_):
    print("Feature:%d, score:%.1f" % (i, v))

k8Best = SelectKBest(mutual_info_regression, k=8)
data_shrink = k8Best.fit_transform(data_train, target_train)
print(k8Best.get_support())
print(data_shrink[:1])
print(data_train[:1])
plt.bar([x for x in range(len(regression1.coef_))], regression1.coef_)
plt.show()