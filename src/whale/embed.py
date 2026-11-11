"""Compute and cache Perch embeddings for every clip of a subset.

Embeddings depend only on the clip and the input variant, not on the fold, so
one pass serves all cross-validation folds. The run is resumable: finished
batches are kept in a memory-mapped file and skipped on restart.

Usage::

    python -m whale.embed train --speed 4 --pad tile
    python -m whale.embed test  --speed 4 --pad tile
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from whale import paths
from whale.data.preprocess import clean_clip
from whale.features.perch_input import PerchInputConfig, make_perch_input
from whale.models.perch_onnx import EMBEDDING_DIM, PerchEmbedder


def embedding_path(subset: str, config: PerchInputConfig) -> Path:
    return paths.CACHE_DIR / "perch" / f"{subset}_{config.name}.npy"


def embed_rows(
    embedder: PerchEmbedder,
    waveforms: np.ndarray,
    lengths: np.ndarray,
    rows: np.ndarray,
    config: PerchInputConfig,
    batch_size: int = 32,
    progress: str | None = None,
) -> np.ndarray:
    """Embeddings [len(rows), 1536] for the given waveform rows."""
    out = np.zeros((len(rows), EMBEDDING_DIM), dtype=np.float32)
    starts = range(0, len(rows), batch_size)
    for start in tqdm(starts, desc=progress, disable=progress is None, mininterval=30):
        batch_rows = rows[start : start + batch_size]
        batch = np.stack([make_perch_input(clean_clip(waveforms[r], lengths[r]), config) for r in batch_rows])
        out[start : start + len(batch_rows)] = embedder.embed(batch)
    return out


def embed_subset(subset: str, config: PerchInputConfig, chunk: int = 1024) -> Path:
    manifest = pd.read_parquet(paths.manifest_path(subset))
    waveforms = np.load(paths.waveform_path(subset), mmap_mode="r")
    lengths = manifest["n_samples"].to_numpy()

    target = embedding_path(subset, config)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial.npy")
    progress_file = target.with_suffix(".progress")
    done = int(progress_file.read_text()) if progress_file.exists() and partial.exists() else 0
    mode = "r+" if done else "w+"
    store = np.lib.format.open_memmap(partial, mode=mode, dtype=np.float32, shape=(len(manifest), EMBEDDING_DIM))

    embedder = PerchEmbedder()
    for start in range(done, len(manifest), chunk):
        rows = np.arange(start, min(start + chunk, len(manifest)))
        store[rows] = embed_rows(embedder, waveforms, lengths, rows, config)
        store.flush()
        progress_file.write_text(str(rows[-1] + 1))
        print(f"{subset} {config.name}: {rows[-1] + 1}/{len(manifest)}", flush=True)

    del store
    partial.replace(target)
    progress_file.unlink(missing_ok=True)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("subset", choices=["train", "test"])
    parser.add_argument("--speed", type=int, default=1)
    parser.add_argument("--pad", choices=["zero", "tile"], default="zero")
    parser.add_argument("--peak", type=float, default=0.25)
    args = parser.parse_args()
    print(embed_subset(args.subset, PerchInputConfig(args.speed, args.pad, args.peak)))
