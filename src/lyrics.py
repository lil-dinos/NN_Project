import numpy as np
import re
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from config import LATENT_DIM, RANDOM_SEED


def preprocess_lyrics(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_tfidf_features(lyrics_list: List[str], max_features: int = 500, n_components: Optional[int] = None) -> Tuple[np.ndarray, TfidfVectorizer]:
    processed = [preprocess_lyrics(text) for text in lyrics_list]
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2), min_df=2)
    features = vectorizer.fit_transform(processed).toarray().astype(np.float32)
    if n_components is not None and n_components < features.shape[1]:
        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
        features = svd.fit_transform(features).astype(np.float32)
    return features, vectorizer


def extract_bow_features(lyrics_list: List[str], max_features: int = 500, n_components: Optional[int] = None) -> Tuple[np.ndarray, CountVectorizer]:
    processed = [preprocess_lyrics(text) for text in lyrics_list]
    vectorizer = CountVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 1), min_df=2)
    features = vectorizer.fit_transform(processed).toarray().astype(np.float32)
    row_sums = features.sum(axis=1, keepdims=True)
    features = features / (row_sums + 1e-8)
    if n_components is not None and n_components < features.shape[1]:
        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
        features = svd.fit_transform(features).astype(np.float32)
    return features, vectorizer


def extract_simple_stats(lyrics_list: List[str]) -> np.ndarray:
    features = []
    for text in lyrics_list:
        words = preprocess_lyrics(text).split()
        lines = text.strip().split('\n')
        word_count = len(words)
        unique_words = len(set(words))
        unique_ratio = unique_words / (word_count + 1e-8)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        line_count = len(lines)
        words_per_line = word_count / (line_count + 1e-8)
        features.append([word_count, unique_ratio, avg_word_len, line_count, words_per_line])
    return np.array(features, dtype=np.float32)


def extract_lyrics_embedding(lyrics_list: List[str], method: str = 'tfidf', n_components: int = LATENT_DIM, include_stats: bool = True) -> np.ndarray:
    if method == 'tfidf':
        main_features, _ = extract_tfidf_features(lyrics_list, max_features=500, n_components=n_components - 5 if include_stats else n_components)
    elif method == 'bow':
        main_features, _ = extract_bow_features(lyrics_list, max_features=500, n_components=n_components - 5 if include_stats else n_components)
    else:
        raise ValueError(f"Unknown method: {method}")
    if include_stats:
        stats = extract_simple_stats(lyrics_list)
        stats = (stats - stats.mean(axis=0)) / (stats.std(axis=0) + 1e-8)
        return np.hstack([main_features, stats]).astype(np.float32)
    return main_features.astype(np.float32)


class LyricsEmbedder:
    def __init__(self, method: str = 'tfidf', n_components: int = LATENT_DIM, include_stats: bool = True):
        self.method = method
        self.n_components = n_components
        self.include_stats = include_stats

    def fit_transform(self, lyrics_list: List[str]) -> np.ndarray:
        return extract_lyrics_embedding(lyrics_list, method=self.method, n_components=self.n_components, include_stats=self.include_stats)


def load_lyrics_for_clips(clip_names: List[str], song_stems: List[str], lyrics_loader_func, use_clip_lyrics: bool = True) -> List[str]:
    from features import load_lyrics_for_clip
    lyrics_list = []
    if use_clip_lyrics:
        for clip_name in clip_names:
            lyrics_list.append(load_lyrics_for_clip(clip_name))
    else:
        lyrics_cache = {}
        for stem in song_stems:
            if stem not in lyrics_cache:
                lyrics_cache[stem] = lyrics_loader_func(stem)
            lyrics_list.append(lyrics_cache[stem])
    return lyrics_list
