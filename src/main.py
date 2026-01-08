import sys
import numpy as np
from pathlib import Path

src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from config import validate_paths, ensure_output_dir, get_device, N_CLUSTERS
from features import build_dataset, standardize_features
from training import train_vae, cluster_latent_space, compute_unsupervised_metrics, compute_supervised_metrics, visualize_clusters, save_cluster_listing, print_evaluation_report


def main():
    print("VAE + KMeans Audio Clustering")
    print("=" * 50)
    
    print("\n[1/6] Validating paths...")
    validate_paths()
    
    print("\n[2/6] Building dataset...")
    X, clip_names, song_stems, genres = build_dataset()
    
    print("\n[3/6] Standardizing features...")
    X_scaled, scaler = standardize_features(X)
    
    print("\n[4/6] Training VAE...")
    model, Z = train_vae(X_scaled, device=get_device(), verbose=True)
    
    print("\n[5/6] Clustering...")
    clusters, kmeans = cluster_latent_space(Z, n_clusters=N_CLUSTERS)
    
    print("\n[6/6] Evaluating...")
    unsup_metrics = compute_unsupervised_metrics(Z, clusters)
    sup_metrics = compute_supervised_metrics(clusters, genres)
    print_evaluation_report(unsup_metrics, sup_metrics, clusters, genres)
    
    output_dir = ensure_output_dir()
    save_cluster_listing(clusters, clip_names, song_stems, genres, output_dir / "cluster_listing.txt")
    visualize_clusters(Z, clusters, output_path=output_dir / "tsne_visualization.png")
    np.save(output_dir / "latent_representations.npy", Z)
    np.save(output_dir / "cluster_assignments.npy", clusters)
    
    print("\nPipeline completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
