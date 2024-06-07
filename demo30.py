from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np

PER_GROUP = 100
X = np.r_[np.random.randn(PER_GROUP, 2) + [2, 2],
          np.random.randn(PER_GROUP, 2) + [0, -2],
          np.random.randn(PER_GROUP, 2) + [-2, 2]]

[plt.scatter(e[0], e[1], c='black', s=7) for e in X]
k = 3
C_x = np.random.uniform(np.min(X[:, 0]), np.max(X[:, 0]), size=k)
C_y = np.random.uniform(np.min(X[:, 1]), np.max(X[:, 1]), size=k)
C = np.array(list(zip(C_x, C_y)), dtype=float)
plt.scatter(C_x, C_y, marker='*', s=100, c='#0599FF')


def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


C_old = np.zeros(C.shape)
clusters = np.zeros(len(X))
delta = dist(C, C_old, None)
print(f"初始值的delta是={delta}")
plt.show()


def plot_kmean(current_cluster, delta):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    fig, ax = plt.subplots()
    for i in range(k):
        pts = np.array([X[j] for j in range(len(X)) if current_cluster[j] == i])
        ax.scatter(pts[:, 0], pts[:, 1], s=7, c=colors[i])
    plt.scatter(C[:, 0], C[:, 1], marker='*', s=100, c='#0599FF')
    plt.title("delta will be:%.4f" % delta)
    plt.show()


while delta != 0:
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    C_old = deepcopy(C)
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    delta = dist(C, C_old, None)
    plot_kmean(clusters, delta)