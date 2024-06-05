from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import svm

iris = datasets.load_iris()
data = iris.data
target = iris.target

r1 = LogisticRegression(max_iter=200)
s1 = svm.SVC(kernel='linear')
s2 = svm.SVC(kernel='poly')
s3 = svm.SVC(kernel='rbf')
classifiers = [r1, s1, s2, s3]
for c in classifiers:
    score = model_selection.cross_val_score(c, data, target, cv=3)
    print(c, score, np.mean(score))