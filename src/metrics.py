import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import Counter
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score


def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    return silhouette_score(X, labels)


def compute_calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    return calinski_harabasz_score(X, labels)


def compute_davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float('inf')
    return davies_bouldin_score(X, labels)


def compute_ari(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    return adjusted_rand_score(true_labels, pred_labels)


def compute_nmi(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    return normalized_mutual_info_score(true_labels, pred_labels)


def compute_cluster_purity(true_labels: np.ndarray, pred_labels: np.ndarray) -> Tuple[float, Dict]:
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    unique_clusters = np.unique(pred_labels)
    total_correct = 0
    cluster_info = {}
    for cluster_id in unique_clusters:
        mask = pred_labels == cluster_id
        cluster_true_labels = true_labels[mask]
        counter = Counter(cluster_true_labels)
        most_common_label, most_common_count = counter.most_common(1)[0]
        cluster_info[int(cluster_id)] = {
            'size': int(np.sum(mask)), 'dominant_label': str(most_common_label),
            'dominant_count': int(most_common_count), 'purity': float(most_common_count / np.sum(mask))}
        total_correct += most_common_count
    return total_correct / len(true_labels), cluster_info


def compute_homogeneity_completeness(true_labels: np.ndarray, pred_labels: np.ndarray) -> Tuple[float, float, float]:
    return homogeneity_score(true_labels, pred_labels), completeness_score(true_labels, pred_labels), v_measure_score(true_labels, pred_labels)


def evaluate_clustering(X: np.ndarray, pred_labels: np.ndarray, true_labels: Optional[np.ndarray] = None, verbose: bool = True) -> Dict:
    metrics = {}
    metrics['silhouette'] = compute_silhouette(X, pred_labels)
    metrics['calinski_harabasz'] = compute_calinski_harabasz(X, pred_labels)
    metrics['davies_bouldin'] = compute_davies_bouldin(X, pred_labels)
    metrics['n_clusters'] = len(np.unique(pred_labels))
    if true_labels is not None:
        metrics['ari'] = compute_ari(true_labels, pred_labels)
        metrics['nmi'] = compute_nmi(true_labels, pred_labels)
        purity, cluster_info = compute_cluster_purity(true_labels, pred_labels)
        metrics['purity'] = purity
        h, c, v = compute_homogeneity_completeness(true_labels, pred_labels)
        metrics['homogeneity'] = h
        metrics['completeness'] = c
        metrics['v_measure'] = v
    if verbose:
        print(f"\nSilhouette: {metrics['silhouette']:.4f}, CH: {metrics['calinski_harabasz']:.2f}, DB: {metrics['davies_bouldin']:.4f}")
        if true_labels is not None:
            print(f"ARI: {metrics['ari']:.4f}, NMI: {metrics['nmi']:.4f}, Purity: {metrics['purity']:.4f}")
    return metrics


def compare_methods(results_dict: Dict[str, Dict], metric_names: Optional[List] = None) -> None:
    if metric_names is None:
        metric_names = ['silhouette', 'ari', 'nmi', 'purity']
    print("\nMETHOD COMPARISON")
    print("-" * 60)
    for method_name, metrics in results_dict.items():
        row = f"{method_name:30}"
        for metric in metric_names:
            if metric in metrics:
                row += f"  {metric}: {metrics[metric]:.4f}"
        print(row)
