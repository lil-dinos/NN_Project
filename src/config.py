import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CLIPS_DIR = BASE_DIR / "songs_clips"
SONGS_DIR = BASE_DIR / "songs"
OUTPUT_DIR = BASE_DIR / "output"

SAMPLE_RATE = 22050
N_MELS = 64
HOP_LENGTH = 512

HIDDEN_DIM = 256
LATENT_DIM = 32
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
BETA = 1.0

N_CLUSTERS = 4
RANDOM_SEED = 42
MAX_CLIPS = None


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_paths():
    errors = []
    if not CLIPS_DIR.exists():
        errors.append(f"Clips directory not found: {CLIPS_DIR}")
    if not SONGS_DIR.exists():
        errors.append(f"Songs directory not found: {SONGS_DIR}")
    if errors:
        raise FileNotFoundError("\n".join(errors))
    return True
