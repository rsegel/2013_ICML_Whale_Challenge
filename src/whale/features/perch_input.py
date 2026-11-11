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
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

PERCH_SAMPLE_RATE = 32_000
PERCH_SAMPLES = 160_000  # 5 s
SOURCE_SAMPLE_RATE = 2_000
FADE_SAMPLES = 320  # 10 ms; avoids a broadband click at the clip edges

VALID_SPEEDS = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class PerchInputConfig:
    speed: int = 1
    pad: str = "zero"
    peak: float = 0.25

    @property
    def name(self) -> str:
        return f"speed{self.speed}_{self.pad}_peak{self.peak:g}"


def make_perch_input(clip: np.ndarray, config: PerchInputConfig = PerchInputConfig()) -> np.ndarray:
    """``clip`` is a cleaned (zero-mean) 2 kHz waveform without padding."""
    if config.speed not in VALID_SPEEDS:
        raise ValueError(f"speed must be one of {VALID_SPEEDS}")
    up = PERCH_SAMPLE_RATE // (SOURCE_SAMPLE_RATE * config.speed)
    audio = clip.astype(np.float64)
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
