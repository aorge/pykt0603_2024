import numpy as np
from sklearn import linear_model, datasets

diabetes = datasets.load_diabetes()
print(type(diabetes))
print(dir(diabetes))
print(diabetes.feature_names)
print(diabetes.data.shape)
print(diabetes.target.shape)
print(diabetes.target)

dataForTest = -50

data_train = diabetes.data[:dataForTest]
target_train = diabetes.target[:dataForTest]
print("[train]train shape:", data_train.shape)
print("[train]target shape:", target_train.shape)

data_test = diabetes.data[dataForTest:]
target_test = diabetes.target[dataForTest:]
print("[test]train shape:", data_test.shape)
print("[test]target shape:", target_test.shape)

regression1 = linear_model.LinearRegression()
regression1.fit(data_train, target_train)
print(regression1.coef_)
print(regression1.intercept_)

print("score:", regression1.score(data_test, target_test))

for i in range(dataForTest, 0):
    dataArray = np.array(data_test[i]).reshape(1, -1)
    print("predict:{:.1f}, real:{}".format(regression1.predict(dataArray)[0], target_test[i]))
mean_square_error = np.mean((regression1.predict(data_test) - target_test) ** 2)
print(mean_square_error)
