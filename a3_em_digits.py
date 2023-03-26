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
from bench_gm import bench_gm

from sklearn.datasets._samples_generator import make_blobs

# dataset

# Generate some data
data, labels = load_digits(return_X_y=True)
(n_samples, n_features), n_digits = data.shape, np.unique(labels).size
print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

n_components = min(10, n_features)
n_init = 2


# Dimensionality Reduction
pca = PCA(n_components=n_components).fit(data)
pca_data = pca.fit_transform(data)

ica = FastICA(n_components=n_components).fit(data)
ica_data = ica.fit_transform(data)

rp = GaussianRandomProjection(n_components=n_components).fit(data)
rp_data = rp.fit_transform(data)

lda = LinearDiscriminantAnalysis(n_components=min(n_digits-1, n_features-1, n_components)).fit(data, labels)
lda_data = lda.fit_transform(data, labels)


# Algorithms
print(82 * "_")
print("init\t\ttime\thomo\tcompl\tv-meas\tARI\tAMI")

gm = GaussianMixture(n_components=n_digits, n_init=n_init, random_state=0)
# vanilla gm
bench_gm(gm=gm, name="EM", data=data, labels=labels)

# EM + PCA
bench_gm(gm=gm, name="EM + PCA", data=pca_data, labels=labels)

# EM + ICA
bench_gm(gm=gm, name="EM + ICA", data=ica_data, labels=labels)

# EM + RP
bench_gm(gm=gm, name="EM + RP", data=rp_data, labels=labels)

# EM + LDA
bench_gm(gm=gm, name="EM + LDA", data=lda_data, labels=labels)
print(82 * "_")
