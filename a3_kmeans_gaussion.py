import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture

from sklearn.datasets._samples_generator import make_blobs
from bench_k_means import bench_k_means


n_samples=4000
n_features=2
centers=3
n_components = min(5, n_features)
cluster_std = 1.0
n_init = 10

data1, labels = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=0)
# transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
transformation = [[0.7, -0.7], [-0.3, 0.8]]
data = np.dot(data1, transformation)  # Anisotropic blobs

(n_samples, n_features), n_digits = data.shape, np.unique(labels).size
print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

# print(data[:,0])
# print(data[:,1])

# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))

# axs[0, 0].scatter(data1[:, 0], data1[:, 1], c=labels)
# axs[0, 0].set_title("Before Transformation")

# axs[0, 1].scatter(data[:, 0], data[:, 1], c=labels)
# axs[0, 1].set_title("After Transformation")

# plt.suptitle("Customized Dataset").set_y(0.95)
# plt.show()


# Dimensionality Reduction
pca = PCA(n_components=n_components).fit(data)
pca_data = pca.fit_transform(data)

ica = FastICA(n_components=n_components).fit(data)
ica_data = ica.fit_transform(data)

rp = GaussianRandomProjection(n_components=n_components).fit(data)
rp_data = rp.fit_transform(data)

lda = LinearDiscriminantAnalysis(n_components=min(centers, n_features)-1).fit(data, labels)
lda_data = lda.fit_transform(data, labels)


# Algorithms
print(82 * "_")
print("init\t\ttime\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

# vanilla kmeans
kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=n_init, random_state=0)
bench_k_means(kmeans=kmeans, name="Vanilla k-means", data=data, labels=labels)

# kmeans + PCA
kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=n_init, random_state=0)
bench_k_means(kmeans=kmeans, name="k-means + PCA", data=pca_data, labels=labels)

# kmeans + ICA
kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=n_init, random_state=0)
bench_k_means(kmeans=kmeans, name="k-means + ICA", data=ica_data, labels=labels)

# kmeans + Random Projection
kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=n_init, random_state=0)
bench_k_means(kmeans=kmeans, name="k-means + RP", data=rp_data, labels=labels)

# kmeans + LDA
kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=n_init, random_state=0)
bench_k_means(kmeans=kmeans, name="k-means + LDA", data=lda_data, labels=labels)
print(82 * "_")


# Visualization
# pca = PCA(n_components=2).fit(data)
# pca_data = pca.fit_transform(data)

# ica = FastICA(n_components=2).fit(data)
# ica_data = ica.fit_transform(data)

# rp = GaussianRandomProjection(n_components=2).fit(data)
# rp_data = rp.fit_transform(data)

# lda = LinearDiscriminantAnalysis(n_components=2).fit(data, labels)
# lda_data = lda.fit_transform(data, labels)



fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))

print(pca_data)
print(pca_data.shape)

axs[0, 0].scatter(pca_data[:, 0], pca_data[:, 1], c=labels)
axs[0, 0].set_title("PCA")

axs[0, 1].scatter(ica_data[:, 0], ica_data[:, 1], c=labels)
axs[0, 1].set_title("ICA")

axs[1, 0].scatter(rp_data[:, 0], rp_data[:, 1], c=labels)
axs[1, 0].set_title("Random Projection")

# axs[1, 1].scatter(lda_data[:, 0], lda_data[:, 1], c=labels)
# axs[1, 1].set_title("LDA")

axs[2, 0].scatter(data[:, 0], data[:, 1], c=labels)
axs[2, 0].set_title("Original")

plt.suptitle("Dimension Reduction").set_y(0.95)
plt.show()


# data = pca_data

kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=n_init, random_state=0)
kmeans.fit(data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.scatter(data[:, 0], data[:, 1], c=labels)
# Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(
#     centroids[:, 0],
#     centroids[:, 1],
#     marker="x",
#     s=169,
#     linewidths=3,
#     color="w",
#     zorder=10,
# )
plt.title(
    "K-means clustering on the Gaussian dataset (PCA-reduced data)"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()