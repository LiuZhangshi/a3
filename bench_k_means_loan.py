from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def bench_k_means(kmeans, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time]
    # results = [name, fit_time, estimator[-1].inertia_]
    n = labels.shape[0]
    pos = 0
    for i in range(n):
        if labels[i] > 0 and estimator[-1].labels_[i] > 0:
            pos += 1
        elif labels[i] <= 0 and estimator[-1].labels_[i] <=0:
            pos += 1
    precision = pos/n
    results += [precision]
    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.3f}"
    )
    print(formatter_result.format(*results))