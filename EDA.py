import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score
)
from sklearn.decomposition import PCA

src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from config import validate_paths, RANDOM_SEED, OUTPUT_DIR
from features import build_dataset, standardize_features


def compute_elbow_metrics(X: np.ndarray, k_range: range) -> dict:
    results = {
        'k_values': list(k_range),
        'inertias': [],
        'silhouettes': [],
        'calinski_harabasz': [],
        'davies_bouldin': []
    }
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(X)
        
        results['inertias'].append(kmeans.inertia_)
        
        if k > 1:
            sil = silhouette_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            db = davies_bouldin_score(X, labels)
        else:
            sil, ch, db = 0, 0, 0
        
        results['silhouettes'].append(sil)
        results['calinski_harabasz'].append(ch)
        results['davies_bouldin'].append(db)
    
    return results


def find_optimal_k(results: dict) -> dict:
    k_values = results['k_values']
    
    best_sil_idx = np.argmax(results['silhouettes'])
    best_k_sil = k_values[best_sil_idx]
    
    best_ch_idx = np.argmax(results['calinski_harabasz'])
    best_k_ch = k_values[best_ch_idx]
    
    best_db_idx = np.argmin(results['davies_bouldin'])
    best_k_db = k_values[best_db_idx]
    
    inertias = np.array(results['inertias'])
    if len(inertias) > 2:
        first_diff = np.diff(inertias)
        second_diff = np.diff(first_diff)
        elbow_idx = np.argmax(second_diff) + 2
        best_k_elbow = k_values[elbow_idx] if elbow_idx < len(k_values) else k_values[len(k_values)//2]
    else:
        best_k_elbow = k_values[len(k_values)//2]
    
    return {
        'silhouette': (best_k_sil, results['silhouettes'][best_sil_idx]),
        'calinski_harabasz': (best_k_ch, results['calinski_harabasz'][best_ch_idx]),
        'davies_bouldin': (best_k_db, results['davies_bouldin'][best_db_idx]),
        'elbow': best_k_elbow
    }


def plot_elbow_analysis(results: dict, optimal: dict, save_path: str = None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    k_values = results['k_values']
    
    ax = axes[0, 0]
    ax.plot(k_values, results['inertias'], 'b-o', linewidth=2, markersize=6)
    ax.axvline(x=optimal['elbow'], color='r', linestyle='--', label=f"Elbow at k={optimal['elbow']}")
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Inertia (Within-cluster SSE)', fontsize=12)
    ax.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(k_values, results['silhouettes'], 'g-o', linewidth=2, markersize=6)
    best_k, best_val = optimal['silhouette']
    ax.axvline(x=best_k, color='r', linestyle='--', label=f"Best k={best_k} (score={best_val:.4f})")
    ax.scatter([best_k], [best_val], color='red', s=150, zorder=5, marker='*')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Score (Higher is Better)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(k_values, results['calinski_harabasz'], 'm-o', linewidth=2, markersize=6)
    best_k, best_val = optimal['calinski_harabasz']
    ax.axvline(x=best_k, color='r', linestyle='--', label=f"Best k={best_k} (score={best_val:.2f})")
    ax.scatter([best_k], [best_val], color='red', s=150, zorder=5, marker='*')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Calinski-Harabasz Index', fontsize=12)
    ax.set_title('Calinski-Harabasz Index (Higher is Better)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(k_values, results['davies_bouldin'], 'c-o', linewidth=2, markersize=6)
    best_k, best_val = optimal['davies_bouldin']
    ax.axvline(x=best_k, color='r', linestyle='--', label=f"Best k={best_k} (score={best_val:.4f})")
    ax.scatter([best_k], [best_val], color='red', s=150, zorder=5, marker='*')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Davies-Bouldin Index', fontsize=12)
    ax.set_title('Davies-Bouldin Index (Lower is Better)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def visualize_clusters_pca(X: np.ndarray, k: int, save_path: str = None):
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.6, s=30)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.title(f'PCA Visualization with k={k} clusters')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def main():
    validate_paths()
    
    X, clip_names, song_stems, genres = build_dataset()
    X_scaled, _ = standardize_features(X)
    
    if X_scaled.shape[1] > 50:
        pca = PCA(n_components=50)
        X_reduced = pca.fit_transform(X_scaled)
    else:
        X_reduced = X_scaled
    
    k_range = range(2, 21)
    results = compute_elbow_metrics(X_reduced, k_range)
    optimal = find_optimal_k(results)
    
    k_values = [optimal['silhouette'][0], optimal['calinski_harabasz'][0], 
                optimal['davies_bouldin'][0], optimal['elbow']]
    consensus_k = int(np.median(k_values))
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    plot_elbow_analysis(results, optimal, str(output_dir / "cluster_size_eda.png"))
    visualize_clusters_pca(X_reduced, consensus_k, str(output_dir / f"pca_k{consensus_k}.png"))
    
    return consensus_k


if __name__ == "__main__":
    main()