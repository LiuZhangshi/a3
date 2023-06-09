from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def bench_gm(gm, name, data, labels):
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
    # estimator = make_pipeline(StandardScaler(), gm).fit(data)
    estimator = make_pipeline(gm).fit(data)
    fit_time = time() - t0
    results = [name, fit_time]
    # results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    gm_predict_labels = estimator[-1].predict(data)
    results += [m(labels, gm_predict_labels) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    # results += [
    #     metrics.silhouette_score(
    #         data,
    #         gm_predict_labels,
    #         metric="euclidean",
    #         sample_size=300,
    #     )
    # ]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))