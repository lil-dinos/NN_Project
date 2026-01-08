# VAE-Based Audio Clustering Project

Unsupervised music clustering using Variational Autoencoders on mel-spectrogram features.

## Project Files

| File | Description |
|------|-------------|
| `config.py` | Configuration parameters (paths, hyperparameters, audio settings) |
| `model.py` | Standard VAE architecture with encoder-decoder and loss function |
| `features.py` | Audio feature extraction (mel-spectrograms) and dataset building |
| `advanced_vae.py` | Beta-VAE and Conditional VAE (CVAE) architectures |
| `conv_model.py` | Convolutional VAE for 2D spectrogram processing |
| `lyrics.py` | Lyrics feature extraction (TF-IDF, BoW) |
| `clustering.py` | Clustering algorithms (K-Means, Agglomerative, DBSCAN, GMM) |
| `baselines.py` | Baseline methods (PCA, Autoencoder, Direct clustering) |
| `metrics.py` | Evaluation metrics (Silhouette, ARI, NMI, Purity) |
| `training.py` | VAE training loop and cluster evaluation utilities |
| `visualizations.py` | Plotting functions (t-SNE, PCA, heatmaps, comparisons) |
| `multimodal.py` | Multi-modal fusion (audio + lyrics + genre) |
| `main.py` | Main execution pipeline |
| `__init__.py` | Package exports |

## Quick Start

```bash
python main.py
```

## Requirements

- torch
- numpy
- librosa
- scikit-learn
- matplotlib

## Parameters

- Sample Rate: 22,050 Hz
- Mel Bands: 64
- Latent Dimension: 32
- Hidden Dimension: 256
- Beta (Beta-VAE): 4.0
