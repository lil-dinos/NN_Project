import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def setup_plot_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['font.size'] = 12


def plot_tsne(X, labels, title="t-SNE Visualization", label_names=None, save_path=None, perplexity=30, figsize=(10, 8)):
    setup_plot_style()
    if X.shape[1] > 2:
        perplexity = min(perplexity, len(X) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_2d = tsne.fit_transform(X)
    else:
        X_2d = X
    fig, ax = plt.subplots(figsize=figsize)
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        name = label_names.get(label, f"Cluster {label}") if label_names else f"Cluster {label}"
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[idx]], label=name, alpha=0.7, s=50)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_pca(X, labels, title="PCA Visualization", label_names=None, save_path=None, figsize=(10, 8)):
    setup_plot_style()
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        explained_var = pca.explained_variance_ratio_
    else:
        X_2d = X
        explained_var = [1.0, 0.0]
    fig, ax = plt.subplots(figsize=figsize)
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        name = label_names.get(label, f"Cluster {label}") if label_names else f"Cluster {label}"
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[idx]], label=name, alpha=0.7, s=50)
    ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_cluster_distribution(labels, title="Cluster Size Distribution", label_names=None, save_path=None, figsize=(10, 6)):
    setup_plot_style()
    unique_labels, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=figsize)
    names = [label_names.get(l, f"Cluster {l}") if label_names else f"Cluster {l}" for l in unique_labels]
    bars = ax.bar(names, counts, color=plt.cm.tab10(np.linspace(0, 1, len(unique_labels))))
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(count), ha='center', va='bottom')
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Samples")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_genre_cluster_heatmap(cluster_labels, genre_labels, title="Genre-Cluster Distribution", save_path=None, figsize=(12, 8)):
    setup_plot_style()
    unique_clusters = np.unique(cluster_labels)
    unique_genres = np.unique(genre_labels)
    matrix = np.zeros((len(unique_genres), len(unique_clusters)))
    for i, genre in enumerate(unique_genres):
        for j, cluster in enumerate(unique_clusters):
            mask = (genre_labels == genre) & (cluster_labels == cluster)
            matrix[i, j] = np.sum(mask)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(unique_clusters)))
    ax.set_xticklabels([f"Cluster {c}" for c in unique_clusters])
    ax.set_yticks(range(len(unique_genres)))
    ax.set_yticklabels(unique_genres)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Genre")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_metrics_comparison(results, metrics=None, title="Method Comparison", save_path=None, figsize=(12, 6)):
    setup_plot_style()
    if metrics is None:
        metrics = ['silhouette', 'ari', 'nmi', 'purity']
    available_metrics = [m for m in metrics if any(m in r for r in results.values())]
    methods = list(results.keys())
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        values = [results[m].get(metric, 0) for m in methods]
        bars = ax.bar(range(len(methods)), values, color=colors)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_title(metric.replace('_', ' ').title())
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def create_all_visualizations(X_latent, cluster_labels, genre_labels, output_dir, method_name="vae"):
    os.makedirs(output_dir, exist_ok=True)
    plot_tsne(X_latent, cluster_labels, title=f"{method_name} - t-SNE", save_path=os.path.join(output_dir, f"{method_name}_tsne.png"))
    plot_pca(X_latent, cluster_labels, title=f"{method_name} - PCA", save_path=os.path.join(output_dir, f"{method_name}_pca.png"))
    plot_cluster_distribution(cluster_labels, title=f"{method_name} - Cluster Sizes", save_path=os.path.join(output_dir, f"{method_name}_dist.png"))
    plot_genre_cluster_heatmap(cluster_labels, genre_labels, title=f"{method_name} - Genre-Cluster", save_path=os.path.join(output_dir, f"{method_name}_heatmap.png"))
    plt.close('all')
