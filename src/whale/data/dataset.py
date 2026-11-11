"""PyTorch dataset over the decoded waveform cache."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from whale import paths
from whale.data.preprocess import clean_clip, fade_edges


class ClipDataset(Dataset):
    """Yields (waveform [4000] float32, label, row index into the manifest).

    Each clip is cleaned (offset and level removed), faded in and out over
    10 ms, and placed in a 2 s buffer of zeros. Without augmentation it is centred. With ``max_shift`` > 0 the
    placement is moved by a random number of samples in [-max_shift, max_shift];
    anything pushed past the edge is cut off, as if the detector had fired
    slightly early or late.
    """

    def __init__(self, manifest: pd.DataFrame, waveforms: np.ndarray, rows: np.ndarray,
                 max_shift: int = 0, seed: int = 0):
        self.rows = np.asarray(rows)
        self.waveforms = waveforms
        self.lengths = manifest["n_samples"].to_numpy()
        self.labels = manifest["label"].to_numpy()
        self.max_shift = max_shift
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = int(self.rows[i])
        clip = fade_edges(clean_clip(self.waveforms[row], self.lengths[row]))
        out = np.zeros(paths.CLIP_SAMPLES, dtype=np.float32)
        start = (paths.CLIP_SAMPLES - len(clip)) // 2
        if self.max_shift:
            start += int(self.rng.integers(-self.max_shift, self.max_shift + 1))
        lo, hi = max(start, 0), min(start + len(clip), paths.CLIP_SAMPLES)
        out[lo:hi] = clip[lo - start : hi - start]
        return torch.from_numpy(out), np.float32(self.labels[row]), row
