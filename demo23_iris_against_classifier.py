from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
data = iris.data
target = iris.target

r1 = LogisticRegression(max_iter=200)
s1 = svm.SVC(kernel='linear')
s2 = svm.SVC(kernel='poly')
s3 = svm.SVC(kernel='rbf')
tree1 = tree.DecisionTreeClassifier()
k1 = KNeighborsClassifier(n_neighbors=2)
k3 = KNeighborsClassifier(n_neighbors=4)
k5 = KNeighborsClassifier(n_neighbors=6)
k7 = KNeighborsClassifier(n_neighbors=8)
nb = GaussianNB()
classifiers = [r1, s1, s2, s3, tree1, k1, k3, k5, k7, nb]
for c in classifiers:
    score = model_selection.cross_val_score(c, data, target, cv=3)
    print(c, -score, np.mean(score))