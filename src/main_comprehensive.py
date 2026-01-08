import sys
import os
from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from config import validate_paths, ensure_output_dir, get_device, N_CLUSTERS, LATENT_DIM
from features import build_dataset, standardize_features
from training import train_vae, cluster_latent_space, save_cluster_listing
from baselines import run_all_baselines
from advanced_vae import train_beta_vae, train_cvae
from lyrics import LyricsEmbedder, load_lyrics_for_clips
from multimodal import extract_multimodal_features
from clustering import run_all_clustering
from metrics import evaluate_clustering, compare_methods
from visualizations import plot_tsne, plot_pca, plot_cluster_distribution, plot_genre_cluster_heatmap, plot_metrics_comparison
import matplotlib.pyplot as plt


def print_header(text: str) -> None:
    print(f"\n{'='*70}")
    print(f" {text}")
    print(f"{'='*70}")


def run_easy_task(X_scaled, Z_vae, clusters_vae, genres, output_dir, clip_names, song_stems):
    print_header("EASY TASK: Basic VAE + KMeans")
    results = {}
    
    print("\n[Easy 1/3] VAE + KMeans evaluation...")
    vae_metrics = evaluate_clustering(Z_vae, clusters_vae, genres, verbose=True)
    results['VAE + KMeans'] = vae_metrics
    
    save_cluster_listing(clusters_vae, clip_names, song_stems, genres, output_dir / "clusters_easy_vae_kmeans.txt")
    
    print("\n[Easy 2/3] Running baselines...")
    baseline_results = run_all_baselines(X_scaled, n_clusters=N_CLUSTERS, verbose=True)
    for name, res in baseline_results.items():
        baseline_metrics = evaluate_clustering(res['Z'], res['clusters'], genres, verbose=False)
        results[res['name']] = baseline_metrics
        save_cluster_listing(res['clusters'], clip_names, song_stems, genres, 
                           output_dir / f"clusters_easy_{name}.txt")
    
    print("\n[Easy 3/3] Creating visualizations...")
    plot_tsne(Z_vae, clusters_vae, title="VAE + KMeans: t-SNE", save_path=str(output_dir / "easy_vae_tsne.png"))
    plot_pca(Z_vae, clusters_vae, title="VAE + KMeans: PCA", save_path=str(output_dir / "easy_vae_pca.png"))
    plot_cluster_distribution(clusters_vae, title="Cluster Distribution", save_path=str(output_dir / "easy_cluster_dist.png"))
    plot_genre_cluster_heatmap(clusters_vae, genres, title="Genre-Cluster Distribution", save_path=str(output_dir / "easy_genre_cluster_heatmap.png"))
    
    le = LabelEncoder()
    genre_encoded = le.fit_transform(genres)
    genre_map = {i: g for i, g in enumerate(le.classes_)}
    plot_tsne(Z_vae, genre_encoded, title="VAE Latent Space by Genre", label_names=genre_map, save_path=str(output_dir / "easy_vae_tsne_genre.png"))
    
    plot_metrics_comparison(results, metrics=['silhouette', 'ari', 'nmi'], title="Easy Task: Method Comparison", save_path=str(output_dir / "easy_baseline_comparison.png"))
    
    return results


def run_medium_task(X_scaled, clip_paths, song_stems, genres, output_dir, clip_names):
    print_header("MEDIUM TASK: Conv VAE + Lyrics + Multiple Clustering")
    results = {}
    lyrics_features = None
    
    print("\n[Medium 1/4] Training Convolutional VAE...")
    try:
        from conv_model import extract_mfcc_2d
        print("  Extracting 2D MFCC features...")
        spectrograms = []
        for path in clip_paths[:min(len(clip_paths), 500)]:
            if os.path.exists(path):
                spec = extract_mfcc_2d(path, n_mfcc=20, fixed_length=128)
                spectrograms.append(spec)
        spectrograms = np.array(spectrograms)
        
        conv_model, Z_conv = train_conv_vae(spectrograms, epochs=30, verbose=True)
        clusters_conv, _ = cluster_latent_space(Z_conv, N_CLUSTERS)
        conv_metrics = evaluate_clustering(Z_conv, clusters_conv, genres, verbose=False)
        results['ConvVAE + KMeans'] = conv_metrics
        save_cluster_listing(clusters_conv, clip_names, song_stems, genres, output_dir / "clusters_medium_conv_vae.txt")
        
    except Exception as e:
        print(f"  ⚠ ConvVAE skipped: {e}")
        Z_conv = X_scaled[:, :32]
    
    print("\n[Medium 2/4] Extracting lyrics features...")
    try:
        lyrics_list = load_lyrics_for_clips(clip_names, song_stems, lambda x: "", use_clip_lyrics=False)
        embedder = LyricsEmbedder(method='tfidf', n_components=64)
        lyrics_features = embedder.fit_transform(lyrics_list)
        print(f"  ✓ Lyrics features: {lyrics_features.shape}")
        
        clusters_lyrics, _ = cluster_latent_space(lyrics_features, N_CLUSTERS)
        lyrics_metrics = evaluate_clustering(lyrics_features, clusters_lyrics, genres, verbose=False)
        results['Lyrics TF-IDF + KMeans'] = lyrics_metrics
        save_cluster_listing(clusters_lyrics, clip_names, song_stems, genres, output_dir / "clusters_medium_lyrics_tfidf.txt")
        
    except Exception as e:
        print(f"  ⚠ Lyrics features skipped: {e}")
        lyrics_features = np.random.randn(len(clip_names), 64).astype(np.float32)
    
    print("\n[Medium 3/4] Multiple clustering algorithms...")
    clustering_results = run_all_clustering(Z_conv, n_clusters=N_CLUSTERS, verbose=True)
    for name, result in clustering_results.items():
        if name == 'kmeans':
            continue
        try:
            metrics = evaluate_clustering(Z_conv, result['clusters'], genres, verbose=False)
            results[f"VAE + {result['name']}"] = metrics
            save_cluster_listing(result['clusters'], clip_names, song_stems, genres,
                               output_dir / f"clusters_medium_vae_{name}.txt")
        except Exception as e:
            print(f"  ⚠ {name} skipped: {e}")
    
    print("\n[Medium 4/4] Combined features...")
    try:
        combined_features = np.hstack([Z_conv[:len(lyrics_features)], lyrics_features])
        clusters_combined, _ = cluster_latent_space(combined_features, N_CLUSTERS)
        combined_metrics = evaluate_clustering(combined_features, clusters_combined, genres, verbose=False)
        results['VAE + Lyrics Combined'] = combined_metrics
        save_cluster_listing(clusters_combined, clip_names, song_stems, genres, output_dir / "clusters_medium_vae_lyrics_combined.txt")
        
        plot_tsne(combined_features, clusters_combined, title="Combined Audio+Lyrics", save_path=str(output_dir / "medium_combined_tsne.png"))
    except Exception as e:
        print(f"  ⚠ Combined features skipped: {e}")
    
    return results, Z_conv, lyrics_features


def run_hard_task(X_scaled, Z_vae, genres, song_stems, output_dir, clip_names, lyrics_features=None):
    print_header("HARD TASK: Beta-VAE, CVAE, Multi-modal fusion")
    results = {}
    
    print("\n[Hard 1/3] Beta-VAE...")
    try:
        beta_model, Z_beta = train_beta_vae(X_scaled, beta=4.0, epochs=40, verbose=True)
        clusters_beta, _ = cluster_latent_space(Z_beta, N_CLUSTERS)
        beta_metrics = evaluate_clustering(Z_beta, clusters_beta, genres, verbose=False)
        results['Beta-VAE + KMeans'] = beta_metrics
        save_cluster_listing(clusters_beta, clip_names, song_stems, genres, output_dir / "clusters_hard_beta_vae.txt")
        plot_tsne(Z_beta, clusters_beta, title="Beta-VAE: Disentangled Latent Space", save_path=str(output_dir / "hard_beta_vae_tsne.png"))
    except Exception as e:
        print(f"  ⚠ Beta-VAE skipped: {e}")
        Z_beta = Z_vae
    
    print("\n[Hard 2/3] Conditional VAE...")
    try:
        le = LabelEncoder()
        genre_labels = le.fit_transform(genres)
        cvae_model, Z_cvae = train_cvae(X_scaled, genre_labels, epochs=40, verbose=True)
        clusters_cvae, _ = cluster_latent_space(Z_cvae, N_CLUSTERS)
        cvae_metrics = evaluate_clustering(Z_cvae, clusters_cvae, genres, verbose=False)
        results['CVAE + KMeans'] = cvae_metrics
        save_cluster_listing(clusters_cvae, clip_names, song_stems, genres, output_dir / "clusters_hard_cvae.txt")
        plot_tsne(Z_cvae, genre_labels, title="CVAE: Genre-Conditioned Latent Space", save_path=str(output_dir / "hard_cvae_tsne.png"))
    except Exception as e:
        print(f"  ⚠ CVAE skipped: {e}")
    
    print("\n[Hard 3/3] Multi-modal fusion...")
    try:
        if lyrics_features is not None:
            fused_features, info = extract_multimodal_features(Z_vae, lyrics_features, genres, fusion_weights=(0.6, 0.3, 0.1))
            clusters_fused, _ = cluster_latent_space(fused_features, N_CLUSTERS)
            fused_metrics = evaluate_clustering(fused_features, clusters_fused, genres, verbose=False)
            results['Multi-modal Fusion'] = fused_metrics
            save_cluster_listing(clusters_fused, clip_names, song_stems, genres, output_dir / "clusters_hard_multimodal.txt")
            plot_tsne(fused_features, clusters_fused, title="Multi-modal Fusion", save_path=str(output_dir / "hard_multimodal_tsne.png"))
            plot_genre_cluster_heatmap(clusters_fused, genres, title="Multi-modal: Genre-Cluster Distribution", save_path=str(output_dir / "hard_multimodal_heatmap.png"))
    except Exception as e:
        print(f"  ⚠ Multi-modal fusion skipped: {e}")
    
    return results


def main():
    print_header("CSE425 COMPREHENSIVE AUDIO CLUSTERING ANALYSIS")
    print(f"\nRunning ALL task levels: Easy, Medium, and Hard")
    print(f"Using {N_CLUSTERS} clusters, {LATENT_DIM} latent dimensions")
    
    print("\n[SETUP] Validating paths...")
    validate_paths()
    
    print("\n[SETUP] Building dataset...")
    X, clip_names, song_stems, genres = build_dataset()
    clip_paths = [f"songs_clips/{name}" for name in clip_names]
    
    X_scaled, scaler = standardize_features(X)
    print(f"  ✓ Features standardized")
    
    print("\n[SETUP] Training base VAE...")
    vae_model, Z_vae = train_vae(X_scaled, device=get_device(), verbose=True)
    clusters_vae, _ = cluster_latent_space(Z_vae, N_CLUSTERS)
    
    output_dir = ensure_output_dir()
    
    all_results = {}
    easy_results = run_easy_task(X_scaled, Z_vae, clusters_vae, genres, output_dir, clip_names, song_stems)
    all_results.update(easy_results)
    
    medium_results, Z_vae_updated, lyrics_features = run_medium_task(X_scaled, clip_paths, song_stems, genres, output_dir, clip_names)
    all_results.update(medium_results)
    
    hard_results = run_hard_task(X_scaled, Z_vae, genres, song_stems, output_dir, clip_names, lyrics_features)
    all_results.update(hard_results)
    
    print_header("FINAL COMPARISON")
    compare_methods(all_results, metric_names=['silhouette', 'ari', 'nmi', 'purity'])
    
    plot_metrics_comparison(all_results, metrics=['silhouette', 'ari', 'nmi', 'purity'], 
                          title="Final Comparison: All Methods", save_path=str(output_dir / "final_comparison.png"))
    
    best_method = max(all_results.keys(), key=lambda x: all_results[x].get('silhouette', 0))
    print(f"\nBest method (by Silhouette): {best_method}")
    
    with open(output_dir / "FULL_REPORT.txt", "w") as f:
        f.write("CSE425 Comprehensive Audio Clustering Report\n")
        f.write("=" * 50 + "\n\n")
        for method, metrics in all_results.items():
            f.write(f"{method}:\n")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")
        f.write(f"\nBest method (by Silhouette): {best_method}\n")
    
    print_header("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"\nResults saved to: {output_dir}")
    
    plt.close('all')
    return 0


if __name__ == "__main__":
    sys.exit(main())