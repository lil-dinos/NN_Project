import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, List
from config import HIDDEN_DIM, LATENT_DIM, EPOCHS, BATCH_SIZE, LEARNING_RATE, N_CLUSTERS, RANDOM_SEED, get_device


class BetaVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, beta: float = 4.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.BatchNorm1d(hidden_dim // 2))
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(), nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim))

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def get_latent(self, x):
        mu, _ = self.encode(x)
        return mu

    def loss(self, x, x_recon, mu, logvar):
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss, recon_loss.detach(), kl_loss.detach()


class ConditionalVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, n_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.class_embedding = nn.Embedding(n_classes, hidden_dim // 4)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + hidden_dim // 4, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim // 4, hidden_dim // 2), nn.ReLU(), nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    def encode(self, x, y):
        y_emb = self.class_embedding(y)
        h = torch.cat([x, y_emb], dim=1)
        h = self.encoder(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        y_emb = self.class_embedding(y)
        h = torch.cat([z, y_emb], dim=1)
        return self.decoder(h)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

    def get_latent(self, x, y):
        mu, _ = self.encode(x, y)
        return mu


def cvae_loss(x, x_recon, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss.detach(), kl_loss.detach()


def train_beta_vae(X, beta=4.0, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, device=None, verbose=True):
    if device is None:
        device = get_device()
    if verbose:
        print(f"Training Beta-VAE (beta={beta}) on {device}")
    dataset = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = BetaVAE(X.shape[1], hidden_dim, latent_dim, beta=beta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            x_recon, mu, logvar = model(batch)
            loss, _, _ = model.loss(batch, x_recon, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch:02d}/{epochs} | Loss: {total_loss/len(dataset):.4f}")
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(device)
        Z = model.get_latent(X_tensor).cpu().numpy()
    return model, Z


def train_cvae(X, labels, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, beta=1.0, device=None, verbose=True):
    if device is None:
        device = get_device()
    n_classes = len(np.unique(labels))
    if verbose:
        print(f"Training Conditional VAE on {device}, {n_classes} classes")
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(labels).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ConditionalVAE(X.shape[1], hidden_dim, latent_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            x_recon, mu, logvar = model(batch_x, batch_y)
            loss, _, _ = cvae_loss(batch_x, x_recon, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch:02d}/{epochs} | Loss: {total_loss/len(dataset):.4f}")
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(device)
        y_tensor = torch.from_numpy(labels).long().to(device)
        Z = model.get_latent(X_tensor, y_tensor).cpu().numpy()
    return model, Z
