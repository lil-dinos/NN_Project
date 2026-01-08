from config import (
    BASE_DIR, CLIPS_DIR, SONGS_DIR, OUTPUT_DIR,
    SAMPLE_RATE, N_MELS, HOP_LENGTH,
    HIDDEN_DIM, LATENT_DIM, EPOCHS, BATCH_SIZE, LEARNING_RATE, BETA,
    N_CLUSTERS, RANDOM_SEED, MAX_CLIPS,
    ensure_output_dir, get_device, validate_paths
)

from features import (
    build_dataset, standardize_features, extract_audio_features,
    clip_to_song_stem, load_genre_for_song_stem, load_lyrics_for_song_stem
)

from model import VAE, vae_loss, create_vae

from training import (
    train_vae, cluster_latent_space,
    compute_unsupervised_metrics, compute_supervised_metrics,
    visualize_clusters, save_cluster_listing
)

from advanced_vae import BetaVAE, ConditionalVAE, train_beta_vae, train_cvae

from conv_model import ConvVAE, train_conv_vae, extract_mfcc_2d

from clustering import (
    kmeans_clustering, agglomerative_clustering,
    dbscan_clustering, gmm_clustering, run_all_clustering
)

from baselines import pca_baseline, autoencoder_baseline, run_all_baselines

from lyrics import extract_tfidf_features, extract_lyrics_embedding, LyricsEmbedder

from metrics import evaluate_clustering, compare_methods

from multimodal import early_fusion, weighted_fusion, MultiModalEmbedder, extract_multimodal_features

from visualizations import (
    plot_tsne, plot_pca, plot_cluster_distribution,
    plot_genre_cluster_heatmap, plot_metrics_comparison, create_all_visualizations
)
