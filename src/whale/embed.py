"""Compute and cache Perch embeddings for every clip of a subset.

Embeddings depend only on the clip and the input variant, not on the fold, so
one pass serves all cross-validation folds. The run is resumable: finished
batches are kept in a memory-mapped file and skipped on restart.

Two outputs:

* pooled (default): ``embedding`` [n, 1536] float32, Perch's mean over all
  16 x 4 spatial cells.
* ``--spatial``: ``spatial_embedding`` restricted to the centre time frames,
  [n, 6, 4, 1536] float16. At speed x4 a 2 s clip fills 10% of the centred 5 s
  input, so most of the 16 frames see only padding; the pooled vector averages
  them in. Pooling over the frames that hold the clip is decided later, in the
  probe, from this cache.

Usage::

    python -m whale.embed train --speed 4 --pad zero
    python -m whale.embed train --speed 4 --pad zero --spatial --device cuda --batch-size 128
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from whale import paths
from whale.data.preprocess import clean_clip
from whale.features.perch_input import PerchInputConfig, make_perch_input
from whale.models.perch_onnx import EMBEDDING_DIM, SPATIAL_SHAPE, PerchEmbedder

# Time frames 5-10 of 16 (each about 0.31 s): the clip sits in frames 6-9 at
# speed x4 (activity peaks in 7-8, measured on training clips), plus one frame
# of margin on each side.
CENTRE_FRAMES = slice(5, 11)
N_CENTRE = CENTRE_FRAMES.stop - CENTRE_FRAMES.start


def embedding_path(subset: str, config: PerchInputConfig, spatial: bool = False) -> Path:
    suffix = "_spatial" if spatial else ""
    return paths.CACHE_DIR / "perch" / f"{subset}_{config.name}{suffix}.npy"


def output_spec(spatial: bool) -> tuple[tuple[int, ...], np.dtype]:
    if spatial:
        return (N_CENTRE, *SPATIAL_SHAPE[1:]), np.dtype(np.float16)
    return (EMBEDDING_DIM,), np.dtype(np.float32)


def embed_rows(
    embedder: PerchEmbedder,
    waveforms: np.ndarray,
    lengths: np.ndarray,
    rows: np.ndarray,
    config: PerchInputConfig,
    batch_size: int = 32,
    progress: str | None = None,
    spatial: bool = False,
) -> np.ndarray:
    """Embeddings for the given waveform rows: [len(rows), 1536], or
    [len(rows), 6, 4, 1536] centre frames when ``spatial``."""
    shape, dtype = output_spec(spatial)
    out = np.zeros((len(rows), *shape), dtype=dtype)
    starts = range(0, len(rows), batch_size)
    for start in tqdm(starts, desc=progress, disable=progress is None, mininterval=30):
        batch_rows = rows[start : start + batch_size]
        batch = np.stack([make_perch_input(clean_clip(waveforms[r], lengths[r]), config) for r in batch_rows])
        if spatial:
            result = embedder.embed_spatial(batch)[:, CENTRE_FRAMES]
        else:
            result = embedder.embed(batch)
        out[start : start + len(batch_rows)] = result
    return out


def embed_subset(
    subset: str,
    config: PerchInputConfig,
    chunk: int = 1024,
    spatial: bool = False,
    device: str = "cpu",
    batch_size: int = 32,
    threads: int | None = None,
) -> Path:
    manifest = pd.read_parquet(paths.manifest_path(subset))
    waveforms = np.load(paths.waveform_path(subset), mmap_mode="r")
    lengths = manifest["n_samples"].to_numpy()

    target = embedding_path(subset, config, spatial)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial.npy")
    progress_file = target.with_suffix(".progress")
    done = int(progress_file.read_text()) if progress_file.exists() and partial.exists() else 0
    mode = "r+" if done else "w+"
    shape, dtype = output_spec(spatial)
    store = np.lib.format.open_memmap(partial, mode=mode, dtype=dtype, shape=(len(manifest), *shape))

    embedder = PerchEmbedder(threads=threads, device=device)
    print(f"providers: {embedder.session.get_providers()}", flush=True)
    for start in range(done, len(manifest), chunk):
        rows = np.arange(start, min(start + chunk, len(manifest)))
        store[rows] = embed_rows(embedder, waveforms, lengths, rows, config, batch_size, spatial=spatial)
        store.flush()
        progress_file.write_text(str(rows[-1] + 1))
        print(f"{subset} {target.stem}: {rows[-1] + 1}/{len(manifest)}", flush=True)

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
    parser.add_argument("--spatial", action="store_true", help="store centre-frame spatial embeddings")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=None, help="ONNX Runtime CPU threads (default: all cores)")
    args = parser.parse_args()
    config = PerchInputConfig(args.speed, args.pad, args.peak)
    print(embed_subset(args.subset, config, spatial=args.spatial, device=args.device, batch_size=args.batch_size, threads=args.threads))
