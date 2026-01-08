import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
import librosa
from config import SAMPLE_RATE, N_MELS, HOP_LENGTH, LATENT_DIM, EPOCHS, BATCH_SIZE, LEARNING_RATE, BETA, get_device


class ConvEncoder(nn.Module):
    def __init__(self, n_mels: int, latent_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.flatten_size = None
        self.fc_mu = None
        self.fc_logvar = None
        self.latent_dim = latent_dim

    def _init_fc(self, x):
        self.flatten_size = x.view(x.size(0), -1).size(1)
        device = x.device
        self.fc_mu = nn.Linear(self.flatten_size, self.latent_dim).to(device)
        self.fc_logvar = nn.Linear(self.flatten_size, self.latent_dim).to(device)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        if self.fc_mu is None:
            self._init_fc(x)
        return self.fc_mu(x), self.fc_logvar(x)


class ConvDecoder(nn.Module):
    def __init__(self, n_mels: int, latent_dim: int, output_shape: Tuple[int, int]):
        super().__init__()
        self.output_shape = output_shape
        h, w = output_shape[0] // 16, output_shape[1] // 16
        self.init_h, self.init_w = max(1, h), max(1, w)
        self.fc = nn.Linear(latent_dim, 256 * self.init_h * self.init_w)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, self.init_h, self.init_w)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = self.deconv4(x)
        return F.interpolate(x, size=self.output_shape, mode='bilinear', align_corners=False)


class ConvVAE(nn.Module):
    def __init__(self, n_mels: int, time_frames: int, latent_dim: int):
        super().__init__()
        self.n_mels = n_mels
        self.time_frames = time_frames
        self.latent_dim = latent_dim
        self.encoder = ConvEncoder(n_mels, latent_dim)
        self.decoder = ConvDecoder(n_mels, latent_dim, (n_mels, time_frames))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def get_latent(self, x):
        mu, _ = self.encoder(x)
        return mu


def extract_mfcc_2d(audio_path, sr=SAMPLE_RATE, n_mfcc=20, hop_length=HOP_LENGTH, fixed_length=128):
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    if mfcc.shape[1] < fixed_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, fixed_length - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :fixed_length]
    return mfcc.astype(np.float32)


def conv_vae_loss(x, x_recon, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss.detach(), kl_loss.detach()


def train_conv_vae(spectrograms, latent_dim=LATENT_DIM, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, beta=BETA, device=None, verbose=True):
    if device is None:
        device = get_device()
    n_samples, n_mels, time_frames = spectrograms.shape
    if verbose:
        print(f"Training ConvVAE on {device}, input: {spectrograms.shape}")
    X = spectrograms[:, np.newaxis, :, :]
    X = (X - X.mean()) / (X.std() + 1e-8)
    dataset = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ConvVAE(n_mels, time_frames, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            x_recon, mu, logvar = model(batch)
            loss, _, _ = conv_vae_loss(batch, x_recon, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch:02d}/{epochs} | Loss: {total_loss/n_samples:.4f}")
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(device)
        Z = model.get_latent(X_tensor).cpu().numpy()
    return model, Z
