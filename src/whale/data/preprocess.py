"""Waveform clean-up shared by every model.

The recordings carry a large constant offset (about -730 on the labelled days,
-490 on the Kaggle test days) and the signal level differs between those two
periods. Both are removed per clip so that neither model can key on them.
"""

import numpy as np


def clean_clip(samples: np.ndarray, n_samples: int) -> np.ndarray:
    """Return the real part of a cached clip as float32 with zero mean and unit
    standard deviation. Padding is not included; the caller decides how to pad.
    """
    clip = np.asarray(samples[: int(n_samples)], dtype=np.float32)
    clip = clip - clip.mean()
    return clip / max(float(clip.std()), 1e-6)


def fade_edges(clip: np.ndarray, n_fade: int = 20) -> np.ndarray:
    """Ramp the first and last ``n_fade`` samples (10 ms at 2 kHz) to zero.

    A clip that starts or stops abruptly next to silence produces a click: a
    bright vertical line across all frequencies in the spectrogram.
    """
    n_fade = min(n_fade, len(clip) // 2)
    if n_fade == 0:
        return clip
    ramp = (0.5 * (1 - np.cos(np.pi * np.arange(n_fade) / n_fade))).astype(clip.dtype)
    out = clip.copy()
    out[:n_fade] *= ramp
    out[-n_fade:] *= ramp[::-1]
    return out


def centre_pad(clip: np.ndarray, length: int) -> np.ndarray:
    """Zero-pad ``clip`` symmetrically to ``length`` samples (crop if longer)."""
    clip = clip[:length]
    out = np.zeros(length, dtype=clip.dtype)
    start = (length - len(clip)) // 2
    out[start : start + len(clip)] = clip
    return out
