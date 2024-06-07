from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

PER_GROUP = 5000
X = np.r_[np.random.randn(PER_GROUP, 2) + [2, 2],
          np.random.randn(PER_GROUP, 2) + [0, -2],
          np.random.randn(PER_GROUP, 2) + [-2, 2]]
inertias = []
for k in range(1, 20):
    print("now process {}".format(k))
    kmeans = KMeans(n_clusters=k, n_init=100)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 20), inertias)
plt.xticks(range(0, 20))
plt.show()
