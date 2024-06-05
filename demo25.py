import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=40, centers=2)
classifier = svm.SVC(kernel='linear')
classifier.fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
ax = plt.gca()
ax.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1],
           s=100, linewidth=1, facecolors='none', edgecolors='k')

xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = classifier.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, levels=[-1, 0, 1], colors='k', alpha=0.5,
           linestyles=['--', '-', '--'])
print(plt.show())