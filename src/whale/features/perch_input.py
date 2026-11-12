"""Turn a 2 kHz clip into the 5 s, 32 kHz waveform Perch v2 expects.

Perch was trained on wideband recordings (mostly birds). Our clips are at most
2 s at 2 kHz, with the upcall between about 80 and 250 Hz. Perch cannot be told
a sample rate: it assumes every sample is 1/32000 s. Three things are chosen:

``speed``
    1 keeps the audio faithful: upsample 16x, the call stays at 80-250 Hz and
    lasts about 1 s, but occupies only the lowest ~8 of Perch's 128 mel bands.
    A value k > 1 plays the clip k times faster by upsampling only 16/k times:
    the call moves up to k * (80-250) Hz and shrinks to 1/k s. 16 means the
    raw samples are passed through untouched.

``pad``
    How to fill the rest of the 5 s: ``"zero"`` centres the clip in silence,
    ``"tile"`` repeats it end to end.

``peak``
    Peak amplitude after scaling. Perch v2's expected level is undocumented;
    its embeddings are stable for peaks between 0.1 and 1.0 (measured).

``denoise``
    The CNN subtracts each frequency's median dB level, which removes the steady
    tonal bands (20-160 Hz) under the call. Perch computes its own spectrogram
    inside the model, so the equivalent has to be done on the waveform: STFT at
    2 kHz with the CNN's settings, change magnitudes, keep phases, inverse STFT.
    ``"none"`` (default) leaves the audio alone. ``"whiten"`` is the direct
    analogue of the CNN step: divide each frequency by its median magnitude
    (subtracting in dB is dividing in magnitude) and drop everything outside the
    CNN's 40-500 Hz crop, so near-empty high bands are not blown up.
    ``"subtract"`` is classic spectral subtraction: take each frequency's median
    power off every frame, floored at -20 dB, which removes steady bands without
    lifting anything else. Both are applied before resampling and are untuned.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import istft, resample_poly, stft

PERCH_SAMPLE_RATE = 32_000
PERCH_SAMPLES = 160_000  # 5 s
SOURCE_SAMPLE_RATE = 2_000
FADE_SAMPLES = 320  # 10 ms; avoids a broadband click at the clip edges

VALID_SPEEDS = (1, 2, 4, 8, 16)
VALID_DENOISE = ("none", "whiten", "subtract")

# Same STFT as the CNN's spectrogram (whale.features.spectrogram).
STFT_WINDOW, STFT_HOP, STFT_NFFT = 256, 16, 512
BAND_HZ = (40.0, 500.0)
SUBTRACT_FLOOR = 0.01  # keep at least 1% of each cell's power (-20 dB)


@dataclass(frozen=True)
class PerchInputConfig:
    speed: int = 1
    pad: str = "zero"
    peak: float = 0.25
    denoise: str = "none"

    @property
    def name(self) -> str:
        # "none" keeps the old name so existing embedding caches still match.
        base = f"speed{self.speed}_{self.pad}_peak{self.peak:g}"
        return base if self.denoise == "none" else f"{base}_{self.denoise}"


def denoise_clip(clip: np.ndarray, method: str) -> np.ndarray:
    """Remove steady per-frequency background from a 2 kHz clip (see module docstring)."""
    if method == "none":
        return clip
    if method not in VALID_DENOISE:
        raise ValueError(f"denoise must be one of {VALID_DENOISE}")
    freqs, _, spec = stft(clip, fs=SOURCE_SAMPLE_RATE, window="hann", nperseg=STFT_WINDOW,
                          noverlap=STFT_WINDOW - STFT_HOP, nfft=STFT_NFFT)
    magnitude = np.abs(spec)
    if method == "whiten":
        gain = 1.0 / (np.median(magnitude, axis=1, keepdims=True) + 1e-12)
        gain = np.where(((freqs >= BAND_HZ[0]) & (freqs < BAND_HZ[1]))[:, None], gain, 0.0)
    else:
        power = magnitude**2
        noise = np.median(power, axis=1, keepdims=True)
        gain = np.sqrt(np.maximum(1.0 - noise / (power + 1e-12), SUBTRACT_FLOOR))
    _, out = istft(spec * gain, fs=SOURCE_SAMPLE_RATE, window="hann", nperseg=STFT_WINDOW,
                   noverlap=STFT_WINDOW - STFT_HOP, nfft=STFT_NFFT)
    return out[: len(clip)]


def make_perch_input(clip: np.ndarray, config: PerchInputConfig = PerchInputConfig()) -> np.ndarray:
    """``clip`` is a cleaned (zero-mean) 2 kHz waveform without padding."""
    if config.speed not in VALID_SPEEDS:
        raise ValueError(f"speed must be one of {VALID_SPEEDS}")
    up = PERCH_SAMPLE_RATE // (SOURCE_SAMPLE_RATE * config.speed)
    audio = denoise_clip(clip.astype(np.float64), config.denoise)
    if up > 1:
        audio = resample_poly(audio, up, 1)
    audio = audio[:PERCH_SAMPLES].copy()

    fade = min(FADE_SAMPLES, len(audio) // 2)
    if fade > 0:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(fade) / fade))
        audio[:fade] *= ramp
        audio[-fade:] *= ramp[::-1]

    if config.pad == "zero":
        out = np.zeros(PERCH_SAMPLES)
        start = (PERCH_SAMPLES - len(audio)) // 2
        out[start : start + len(audio)] = audio
    elif config.pad == "tile":
        out = np.resize(audio, PERCH_SAMPLES)
    else:
        raise ValueError("pad must be 'zero' or 'tile'")

    peak = np.abs(out).max()
    return (out * (config.peak / peak) if peak > 0 else out).astype(np.float32)
