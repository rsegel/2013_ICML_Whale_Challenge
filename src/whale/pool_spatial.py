"""Pool the cached centre-frame spatial embeddings into 1536-d variants.

``whale.embed --spatial`` stores Perch's spatial embedding for time frames
5-10 of 16 (``CENTRE_FRAMES``), all 4 frequency bands. Perch's own pooled
embedding averages all 16 x 4 cells, most of which see only padding. Variants:

* ``centre4``  mean over frames 6-9 and all bands (the frames that hold a 2 s clip)
* ``centre6``  mean over frames 5-10 and all bands (one frame of margin each side)
* ``tmax6``    mean over bands, then max over frames 5-10

Each is written next to the pooled file as ``<subset>_<input>_<variant>.npy``
[n, 1536] float32, and read by ``whale.probe --features <variant>``. Needs
numpy only, so it runs in a minimal environment.

Usage::

    python -m whale.pool_spatial train --speed 4 --pad zero
"""

import argparse

import numpy as np

from whale import paths
from whale.embed import CENTRE_FRAMES, embedding_path
from whale.features.perch_input import PerchInputConfig

# Positions inside the stored block (stored frame 0 = original frame 5).
_C4 = slice(6 - CENTRE_FRAMES.start, 10 - CENTRE_FRAMES.start)

VARIANTS = {
    "centre4": lambda s: s[:, _C4].mean(axis=(1, 2)),
    "centre6": lambda s: s.mean(axis=(1, 2)),
    "tmax6": lambda s: s.mean(axis=2).max(axis=1),
}


def variant_path(subset: str, config: PerchInputConfig, variant: str):
    return paths.CACHE_DIR / "perch" / f"{subset}_{config.name}_{variant}.npy"


def pool(subset: str, config: PerchInputConfig, chunk: int = 2048) -> None:
    spatial = np.load(embedding_path(subset, config, spatial=True), mmap_mode="r")
    n = len(spatial)
    out = {v: np.zeros((n, spatial.shape[-1]), np.float32) for v in VARIANTS}
    activity = np.zeros(spatial.shape[1], np.float64)  # mean |value| per stored frame
    for start in range(0, n, chunk):
        block = np.asarray(spatial[start : start + chunk], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError(f"non-finite values in rows {start}-{start + len(block)}")
        activity += np.abs(block).mean(axis=(2, 3)).sum(axis=0)
        for v, f in VARIANTS.items():
            out[v][start : start + len(block)] = f(block)
        print(f"{start + len(block)}/{n}", flush=True)

    frames = range(CENTRE_FRAMES.start, CENTRE_FRAMES.stop)
    print("mean |activation| per time frame:",
          ", ".join(f"{f}: {a:.3f}" for f, a in zip(frames, activity / n)))
    for v, x in out.items():
        target = variant_path(subset, config, v)
        np.save(target, x)
        print(f"{v}: {x.shape} -> {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("subset", choices=["train", "test"])
    parser.add_argument("--speed", type=int, default=4)
    parser.add_argument("--pad", choices=["zero", "tile"], default="zero")
    parser.add_argument("--peak", type=float, default=0.25)
    args = parser.parse_args()
    pool(args.subset, PerchInputConfig(args.speed, args.pad, args.peak))
