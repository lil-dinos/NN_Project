import os
import re
import glob
import numpy as np
import librosa
from pathlib import Path
from typing import List, Tuple, Optional
from config import CLIPS_DIR, SONGS_DIR, SAMPLE_RATE, N_MELS, HOP_LENGTH, MAX_CLIPS


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def normalize_genre(genre_str: str) -> str:
    return re.sub(r"\s+", " ", genre_str.strip()).lower()


def clip_to_song_stem(clip_filename: str) -> str:
    base = os.path.splitext(clip_filename)[0]
    base = re.sub(r'_(\d{3}|rest)$', '', base)
    return base


def load_genre_for_song_stem(song_stem: str, songs_dir: Path = SONGS_DIR) -> str:
    genre_path = CLIPS_DIR / f"{song_stem}.genre"
    if genre_path.exists():
        return normalize_genre(read_text(str(genre_path)))
    genre_path = songs_dir / f"{song_stem}.genre"
    if genre_path.exists():
        return normalize_genre(read_text(str(genre_path)))
    return "unknown"


def load_lyrics_for_song_stem(song_stem: str, songs_dir: Path = SONGS_DIR) -> str:
    lyrics_path = CLIPS_DIR / f"{song_stem}.txt"
    if lyrics_path.exists():
        return read_text(str(lyrics_path))
    lyrics_path = songs_dir / f"{song_stem}.txt"
    if lyrics_path.exists():
        return read_text(str(lyrics_path))
    return ""


def load_lyrics_for_clip(clip_filename: str, clips_dir: Path = CLIPS_DIR, songs_dir: Path = SONGS_DIR) -> str:
    base = os.path.splitext(clip_filename)[0]
    match = re.search(r'_(\d{3})$', base)
    if match:
        timestamp = match.group(1)
        song_stem = base.rsplit('_', 1)[0]
        lyrics_filename = f"{song_stem}_lyrics_{timestamp}.txt"
        lyrics_path = clips_dir / lyrics_filename
        if lyrics_path.exists():
            return read_text(str(lyrics_path))
    song_stem = clip_to_song_stem(clip_filename)
    return load_lyrics_for_song_stem(song_stem, songs_dir)


def load_lyrics(song_stem: str, songs_dir: Path = SONGS_DIR) -> str:
    return load_lyrics_for_song_stem(song_stem, songs_dir)


def extract_audio_features(mp3_path: str) -> np.ndarray:
    y, _ = librosa.load(mp3_path, sr=SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel + 1e-9)
    mean = log_mel.mean(axis=1)
    std = log_mel.std(axis=1)
    return np.concatenate([mean, std], axis=0).astype(np.float32)


def build_dataset(clips_dir: Path = CLIPS_DIR, songs_dir: Path = SONGS_DIR, max_clips: Optional[int] = MAX_CLIPS, verbose: bool = True) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    clips = sorted(glob.glob(str(clips_dir / "*.mp3")))
    if max_clips is not None:
        clips = clips[:max_clips]
    if verbose:
        print(f"Found {len(clips)} clips to process")
    if len(clips) == 0:
        raise RuntimeError(f"No mp3 clips found in {clips_dir}")
    X, clip_names, song_stems, genres = [], [], [], []
    for i, mp3_path in enumerate(clips, start=1):
        clip_name = os.path.basename(mp3_path)
        stem = clip_to_song_stem(clip_name)
        genre = load_genre_for_song_stem(stem, songs_dir)
        try:
            features = extract_audio_features(mp3_path)
        except Exception as e:
            if verbose:
                print(f"Failed to process {clip_name}: {e}")
            continue
        X.append(features)
        clip_names.append(clip_name)
        song_stems.append(stem)
        genres.append(genre)
        if verbose and i % 200 == 0:
            print(f"  Processed {i}/{len(clips)}")
    X = np.vstack(X).astype(np.float32)
    if verbose:
        print(f"\nDataset built: {X.shape[0]} samples, {len(set(genres))} genres")
    return X, clip_names, song_stems, genres


def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, object]:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    return X_scaled, scaler
