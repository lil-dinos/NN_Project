import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Tuple, Optional
from config import N_CLUSTERS, RANDOM_SEED


def kmeans_clustering(Z: np.ndarray, n_clusters: int = N_CLUSTERS, random_state: int = RANDOM_SEED) -> Tuple[np.ndarray, KMeans]:
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    clusters = model.fit_predict(Z)
    return clusters, model


def agglomerative_clustering(Z: np.ndarray, n_clusters: int = N_CLUSTERS, linkage: str = 'ward') -> Tuple[np.ndarray, AgglomerativeClustering]:
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    clusters = model.fit_predict(Z)
    return clusters, model


def estimate_dbscan_eps(Z: np.ndarray, k: int = 5) -> float:
    nbrs = NearestNeighbors(n_neighbors=k).fit(Z)
    distances, _ = nbrs.kneighbors(Z)
    k_distances = np.sort(distances[:, k-1])
    return np.percentile(k_distances, 90)


def dbscan_clustering(Z: np.ndarray, eps: Optional[float] = None, min_samples: int = 5) -> Tuple[np.ndarray, DBSCAN]:
    if eps is None:
        eps = estimate_dbscan_eps(Z, k=min_samples)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = model.fit_predict(Z)
    return clusters, model


def gmm_clustering(Z: np.ndarray, n_clusters: int = N_CLUSTERS, random_state: int = RANDOM_SEED) -> Tuple[np.ndarray, GaussianMixture]:
    model = GaussianMixture(n_components=n_clusters, random_state=random_state, covariance_type='full')
    model.fit(Z)
    clusters = model.predict(Z)
    return clusters, model


def run_all_clustering(Z: np.ndarray, n_clusters: int = N_CLUSTERS, verbose: bool = True) -> Dict[str, dict]:
    results = {}
    if verbose:
        print("\nRunning Multiple Clustering Algorithms")
    clusters_km, model_km = kmeans_clustering(Z, n_clusters)
    results['kmeans'] = {'clusters': clusters_km, 'model': model_km, 'name': 'K-Means', 'n_clusters': n_clusters}
    clusters_agg, model_agg = agglomerative_clustering(Z, n_clusters)
    results['agglomerative'] = {'clusters': clusters_agg, 'model': model_agg, 'name': 'Agglomerative (Ward)', 'n_clusters': n_clusters}
    clusters_db, model_db = dbscan_clustering(Z)
    n_clusters_db = len(set(clusters_db)) - (1 if -1 in clusters_db else 0)
    results['dbscan'] = {'clusters': clusters_db, 'model': model_db, 'name': 'DBSCAN', 'n_clusters': n_clusters_db}
    clusters_gmm, model_gmm = gmm_clustering(Z, n_clusters)
    results['gmm'] = {'clusters': clusters_gmm, 'model': model_gmm, 'name': 'Gaussian Mixture', 'n_clusters': n_clusters}
    return results


def get_cluster_distribution(clusters: np.ndarray) -> Dict[int, int]:
    unique, counts = np.unique(clusters, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))
