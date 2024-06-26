from sklearn import tree
from matplotlib import pyplot

X = [[0, 0], [1, 1]]
Y = [0, 1]

classifier1 = tree.DecisionTreeClassifier()
classifier1.fit(X, Y)
print(classifier1)
print(classifier1.tree_)

NewX = [[5, 0], [0, 5], [-5, 0], [0, -5], [5, 5], [5, -5], [-5, -5], [-5, 5]]
print(classifier1.predict(NewX))
tree.plot_tree(classifier1)
pyplot.show()