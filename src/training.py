import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from collections import defaultdict
from pathlib import Path
from model import VAE, vae_loss, create_vae
from config import HIDDEN_DIM, LATENT_DIM, EPOCHS, BATCH_SIZE, LEARNING_RATE, BETA, N_CLUSTERS, RANDOM_SEED, OUTPUT_DIR, ensure_output_dir, get_device


def train_vae(X, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, beta=BETA, device=None, verbose=True):
    if device is None:
        device = get_device()
    if verbose:
        print(f"Training VAE on {device}")
    dataset = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = create_vae(X.shape[1], hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            x_hat, mu, logvar = model(batch)
            loss, recon, kl = vae_loss(batch, x_hat, mu, logvar, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        if verbose:
            print(f"Epoch {epoch:02d}/{epochs} | Loss: {total_loss/len(dataset):.4f}")
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(device)
        Z = model.get_latent(X_tensor).cpu().numpy()
    return model, Z


def cluster_latent_space(Z, n_clusters=N_CLUSTERS, random_state=RANDOM_SEED):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    clusters = kmeans.fit_predict(Z)
    return clusters, kmeans


def compute_unsupervised_metrics(Z, clusters):
    return {
        "silhouette": silhouette_score(Z, clusters),
        "calinski_harabasz": calinski_harabasz_score(Z, clusters),
        "davies_bouldin": davies_bouldin_score(Z, clusters)}


def compute_supervised_metrics(clusters, genres):
    unique_genres = sorted(list(set(genres)))
    genre_map = {g: i for i, g in enumerate(unique_genres)}
    y_true = np.array([genre_map[g] for g in genres], dtype=int)
    return {
        "adjusted_rand_index": adjusted_rand_score(y_true, clusters),
        "normalized_mutual_info": normalized_mutual_info_score(y_true, clusters)}


def visualize_clusters(Z, clusters, output_path=None, title="t-SNE of VAE Latent Space"):
    perplexity = min(30, max(5, len(Z) // 200))
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=perplexity)
    Z_2d = tsne.fit_transform(Z)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c=clusters, cmap='tab10', s=8, alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()


def save_cluster_listing(clusters, clip_names, song_stems, genres, output_path):
    cluster_to_items = defaultdict(list)
    for clip_name, stem, genre, cluster in zip(clip_names, song_stems, genres, clusters):
        cluster_to_items[int(cluster)].append((clip_name, stem, genre))
    with open(output_path, "w", encoding="utf-8") as f:
        for k in sorted(cluster_to_items.keys()):
            items = cluster_to_items[k]
            f.write(f"\nCLUSTER {k} ({len(items)} clips)\n")
            genre_counts = defaultdict(int)
            for _, _, g in items:
                genre_counts[g] += 1
            f.write("Genre distribution:\n")
            for g, count in sorted(genre_counts.items(), key=lambda x: -x[1]):
                f.write(f"  {g}: {count} clips\n")
            f.write("\nClips:\n")
            for clip_name, stem, g in items:
                f.write(f"  {clip_name} | {g}\n")


def print_evaluation_report(unsup_metrics, sup_metrics, clusters, genres):
    print("\nEVALUATION REPORT")
    print(f"Silhouette: {unsup_metrics['silhouette']:.4f}")
    print(f"Calinski-Harabasz: {unsup_metrics['calinski_harabasz']:.4f}")
    print(f"Davies-Bouldin: {unsup_metrics['davies_bouldin']:.4f}")
    print(f"Adjusted Rand Index: {sup_metrics['adjusted_rand_index']:.4f}")
    print(f"Normalized Mutual Info: {sup_metrics['normalized_mutual_info']:.4f}")
    print(f"Clusters: {len(set(clusters))}, Genres: {len(set(genres))}")
