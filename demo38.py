from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib

# pip install matplotlib==3.5.3
print(matplotlib.__version__)

fig = plt.figure(1, figsize=(8, 8))
#ax = Axes3D(fig, elev=-150, azim=110)
ax = fig.add_subplot(projection="3d", elev=-150, azim=110) # matplotlib 3.9

iris = datasets.load_iris()
X = iris.data
species = iris.target
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(iris.data)
print(pca.explained_variance_ratio_)

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=species, cmap=plt.cm.Paired)
ax.set_xlabel("first eigenvalue")
ax.set_ylabel("second eigenvalue")
ax.set_zlabel("third eigenvalue")
plt.show()