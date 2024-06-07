from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

K = 4
PER_GROUP = 100
X = np.r_[np.random.randn(PER_GROUP, 2) + [2, 2],
          np.random.randn(PER_GROUP, 2) + [0, -2],
          np.random.randn(PER_GROUP, 2) + [-2, 2]]

kmeans = KMeans(n_clusters=K, n_init=100)
kmeans.fit(X)

centers = kmeans.cluster_centers_
inertia = kmeans.inertia_

colors = ['c', 'm', 'y', 'k']
markers = ['o', 'v', '*', 'x']
for i in range(K):
    dataX = X[kmeans.labels_ == i]
    plt.scatter(dataX[:, 0], dataX[:, 1], c=colors[i], marker=markers[i])
plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=200, c='#0599FF')
plt.title('inertia={}'.format(inertia))
plt.show()
