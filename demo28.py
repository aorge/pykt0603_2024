import graphviz as gv
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.tree import export_graphviz
from subprocess import check_call

print("graphviz version={}".format(gv.__version__))

X = [[0, 0], [1, 1], [0, 1], [1, 0]]
Y = [0, 0, 1, 1]
col = ['red', 'green']
marker = ['o', 'd']
index = 0
while index < len(X):
    type = Y[index]
    plt.scatter(X[index][0], X[index][1], c=col[type], marker=marker[type])
    index += 1

classifier1 = tree.DecisionTreeClassifier()
classifier1.fit(X, Y)
# 手動先建目錄output
OUTPUT_DOT = "output/demo28.dot"
OUTPUT_PNG = "output/demo28.png"
export_graphviz(classifier1, out_file=OUTPUT_DOT, filled=True, rounded=True, special_characters=True)

check_call(['dot', '-Tpng', OUTPUT_DOT, '-o', OUTPUT_PNG])
newX = [[5, 0], [5, 5], [0, 5], [0, -5], [-5, -5]]
print(classifier1.predict(newX))
plt.show()
