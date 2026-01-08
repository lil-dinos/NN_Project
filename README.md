# VAE Audio Clustering 

## Overview
VAE-based audio clustering project for music genre analysis using mel-spectrogram features.

## Requirements
```bash
pip install -r requirements.txt
```

## Run Instructions

### 1. Full Pipeline
```bash
cd src
python main_comprehensive.py
```

### 2. Cluster Size EDA (Optional)
```bash
python EDA.py
```

### 3. Basic Pipeline
```bash
cd src
python main.py
```

## Configuration
- Edit `src/config.py` to adjust paths and parameters
- Default: 32D latent space, 64 mel bands, 22kHz audio

## Results Location
All outputs saved to `output/` folder:
- **Visualizations**: t-SNE/PCA plots, heatmaps, comparisons
- **Clustering results**: Saved in txt files
- **Metrics**: JSON format with evaluation scores
