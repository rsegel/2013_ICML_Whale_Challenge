"""Locations of raw data and derived caches.

Raw audio stays where Kaggle unpacked it. Everything derived (decoded
waveforms, embeddings, checkpoints) goes to a cache directory that is kept
out of the repository and out of any cloud-synced folder.

Override either location with the environment variables below.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("WHALE_DATA_DIR", REPO_ROOT.parent))
CACHE_DIR = Path(os.environ.get("WHALE_CACHE_DIR", Path.home() / "dv_cache"))

TRAIN_AUDIO_DIR = DATA_DIR / "train2"
TEST_AUDIO_DIR = DATA_DIR / "test2"

SAMPLE_RATE = 2000
CLIP_SAMPLES = 4000  # 2 s at 2 kHz


def waveform_path(subset: str) -> Path:
    return CACHE_DIR / f"{subset}_waveforms.npy"


def manifest_path(subset: str) -> Path:
    return CACHE_DIR / f"{subset}_manifest.parquet"
