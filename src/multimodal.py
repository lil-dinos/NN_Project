import numpy as np
from typing import Tuple, Optional, Dict, List
from sklearn.preprocessing import StandardScaler, LabelEncoder


def normalize_features(X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def early_fusion(audio_features, lyrics_features, genre_labels=None, weights=(1.0, 1.0, 0.5)):
    audio_norm = normalize_features(audio_features) * weights[0]
    lyrics_norm = normalize_features(lyrics_features) * weights[1]
    features_list = [audio_norm, lyrics_norm]
    if genre_labels is not None:
        le = LabelEncoder()
        genre_encoded = le.fit_transform(genre_labels)
        n_classes = len(le.classes_)
        genre_onehot = np.zeros((len(genre_labels), n_classes))
        genre_onehot[np.arange(len(genre_labels)), genre_encoded] = weights[2]
        features_list.append(genre_onehot)
    return np.concatenate(features_list, axis=1)


def weighted_fusion(audio_features, lyrics_features, audio_weight=0.6, lyrics_weight=0.4, target_dim=None):
    from sklearn.decomposition import PCA
    audio_norm = normalize_features(audio_features)
    lyrics_norm = normalize_features(lyrics_features)
    if audio_norm.shape[1] != lyrics_norm.shape[1]:
        target = min(audio_norm.shape[1], lyrics_norm.shape[1], target_dim or 64)
        if audio_norm.shape[1] > target:
            audio_norm = PCA(n_components=target).fit_transform(audio_norm)
        if lyrics_norm.shape[1] > target:
            lyrics_norm = PCA(n_components=target).fit_transform(lyrics_norm)
    return audio_weight * audio_norm + lyrics_weight * lyrics_norm


class MultiModalEmbedder:
    def __init__(self, audio_weight=0.6, lyrics_weight=0.3, genre_weight=0.1, fusion_type='early'):
        self.audio_weight = audio_weight
        self.lyrics_weight = lyrics_weight
        self.genre_weight = genre_weight
        self.fusion_type = fusion_type
        self.audio_scaler = StandardScaler()
        self.lyrics_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit_transform(self, audio_features, lyrics_features, genre_labels=None):
        audio_norm = self.audio_scaler.fit_transform(audio_features)
        lyrics_norm = self.lyrics_scaler.fit_transform(lyrics_features)
        audio_weighted = audio_norm * self.audio_weight
        lyrics_weighted = lyrics_norm * self.lyrics_weight
        if genre_labels is not None:
            genre_encoded = self.label_encoder.fit_transform(genre_labels)
            n_classes = len(self.label_encoder.classes_)
            genre_onehot = np.zeros((len(genre_labels), n_classes))
            genre_onehot[np.arange(len(genre_labels)), genre_encoded] = self.genre_weight
            embeddings = np.concatenate([audio_weighted, lyrics_weighted, genre_onehot], axis=1)
        else:
            embeddings = np.concatenate([audio_weighted, lyrics_weighted], axis=1)
        self.is_fitted = True
        return embeddings


def extract_multimodal_features(audio_features, lyrics_features, genre_labels=None, fusion_weights=(0.6, 0.3, 0.1)):
    embedder = MultiModalEmbedder(audio_weight=fusion_weights[0], lyrics_weight=fusion_weights[1], genre_weight=fusion_weights[2])
    fused_features = embedder.fit_transform(audio_features, lyrics_features, genre_labels)
    info = {'audio_dim': audio_features.shape[1], 'lyrics_dim': lyrics_features.shape[1], 'fused_dim': fused_features.shape[1], 'weights': fusion_weights}
    return fused_features, info
