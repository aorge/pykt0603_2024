from sklearn import linear_model

features = [[0, 1], [1, 3], [2, 8], [3, 20]]
values = [1, 4, 5.5, 9]
# features = [[0, 1], [1, 3], [2, 8]]
# values = [1, 4, 5.5]
r1 = linear_model.LinearRegression()
r1.fit(features, values)

print(r1.coef_)
print(r1.intercept_)
features2 = [[0.8, 0.8], [1, 5], [5, 1], [10, 14]]
result2 = r1.predict(features2)
print(result2)
print(r1.score(features2, result2))
print(r1.score(features, values))