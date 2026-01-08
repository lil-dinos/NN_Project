import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Tuple
from config import HIDDEN_DIM, LATENT_DIM, EPOCHS, BATCH_SIZE, LEARNING_RATE, N_CLUSTERS, RANDOM_SEED, get_device


def pca_baseline(X: np.ndarray, n_components: int = LATENT_DIM, n_clusters: int = N_CLUSTERS, random_state: int = RANDOM_SEED) -> Tuple[np.ndarray, np.ndarray, PCA]:
    pca = PCA(n_components=n_components, random_state=random_state)
    Z_pca = pca.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    clusters = kmeans.fit_predict(Z_pca)
    return Z_pca, clusters, pca


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def train_autoencoder(X, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, device=None, verbose=True):
    if device is None:
        device = get_device()
    if verbose:
        print(f"Training Autoencoder on {device}")
    dataset = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = Autoencoder(X.shape[1], hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            x_recon, _ = model(batch)
            loss = F.mse_loss(x_recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch:02d}/{epochs} | Loss: {total_loss/len(dataset):.4f}")
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(device)
        Z = model.encode(X_tensor).cpu().numpy()
    return model, Z


def autoencoder_baseline(X, n_clusters=N_CLUSTERS, verbose=True):
    model, Z = train_autoencoder(X, verbose=verbose)
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init="auto")
    clusters = kmeans.fit_predict(Z)
    return Z, clusters, model


def direct_spectral_baseline(X, n_clusters=N_CLUSTERS, random_state=RANDOM_SEED):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    return kmeans.fit_predict(X)


def run_all_baselines(X, n_clusters=N_CLUSTERS, verbose=True):
    results = {}
    if verbose:
        print("\nRunning Baseline Methods")
    Z_pca, clusters_pca, pca_model = pca_baseline(X, n_clusters=n_clusters)
    results['pca'] = {'Z': Z_pca, 'clusters': clusters_pca, 'model': pca_model, 'name': 'PCA + K-Means'}
    Z_ae, clusters_ae, ae_model = autoencoder_baseline(X, n_clusters=n_clusters, verbose=verbose)
    results['autoencoder'] = {'Z': Z_ae, 'clusters': clusters_ae, 'model': ae_model, 'name': 'Autoencoder + K-Means'}
    clusters_direct = direct_spectral_baseline(X, n_clusters=n_clusters)
    results['direct'] = {'Z': X, 'clusters': clusters_direct, 'model': None, 'name': 'Direct K-Means'}
    return results
