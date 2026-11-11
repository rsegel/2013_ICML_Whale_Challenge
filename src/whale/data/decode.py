"""Decode every AIFF clip once into a single array.

Reading 70k small files on every epoch is slow, so this writes

* ``<subset>_waveforms.npy``  int16, shape [n_clips, 4000], left-aligned and
  zero-filled (the raw samples, unmodified)
* ``<subset>_manifest.parquet``  one row per clip, row i describes waveform i

Clips shorter than 2 s keep their true length in ``n_samples``; how they are
padded is decided at load time (see ``whale.data.preprocess``), so the cache is
lossless.

Usage::

    python -m whale.data.decode train
    python -m whale.data.decode test
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from whale import paths
from whale.data.manifest import build_manifest


def read_clip(path: Path) -> tuple[np.ndarray, int, int]:
    """Return (int16 samples, sample rate, channel count) for one file."""
    samples, rate = sf.read(path, dtype="int16", always_2d=True)
    return samples[:, 0], rate, samples.shape[1]


def decode_subset(subset: str, workers: int = 8) -> None:
    audio_dir = {"train": paths.TRAIN_AUDIO_DIR, "test": paths.TEST_AUDIO_DIR}[subset]
    manifest = build_manifest(audio_dir)
    files = [audio_dir / name for name in manifest["filename"]]

    waveforms = np.zeros((len(files), paths.CLIP_SAMPLES), dtype=np.int16)
    lengths = np.zeros(len(files), dtype=np.int32)
    rates = np.zeros(len(files), dtype=np.int32)
    channels = np.zeros(len(files), dtype=np.int8)

    with ThreadPoolExecutor(workers) as pool:
        results = pool.map(read_clip, files)
        for i, (samples, rate, n_channels) in enumerate(tqdm(results, total=len(files), desc=subset)):
            n = min(len(samples), paths.CLIP_SAMPLES)
            waveforms[i, :n] = samples[:n]
            lengths[i] = len(samples)  # true length, even if longer than 2 s
            rates[i] = rate
            channels[i] = n_channels

    manifest["n_samples"] = lengths
    manifest["sample_rate"] = rates
    manifest["n_channels"] = channels

    paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(paths.waveform_path(subset), waveforms)
    manifest.to_parquet(paths.manifest_path(subset), index=False)
    print(f"{subset}: {len(files)} clips -> {paths.waveform_path(subset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("subset", choices=["train", "test"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    decode_subset(args.subset, args.workers)
