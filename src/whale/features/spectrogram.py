"""Waveform -> spectrogram image for the CNN baseline.

Computed inside the model's forward pass (not cached) so that augmentation can
act on the waveform and the settings live in one place.

Choices, each backed by the data audit:

* Native 2 kHz, no resampling. Window 256 samples (128 ms), hop 16 (8 ms),
  FFT zero-padded to 512 so frequency bins are 3.9 Hz apart. A 2 s clip gives
  251 frames.
* Keep 40-500 Hz (118 bins). The upcall sweeps about 80 -> 250 Hz; below 40 Hz
  is flow noise and above 500 Hz there is nothing.
* Decibels, then subtract each frequency row's median over time. The
  recordings contain steady tonal bands (20-160 Hz) that overlap the call; a
  steady band has a constant row, so this removes it and leaves transients.
* Clamp to [-10, +30] dB around that median and scale to roughly [-1, 3].
  Zero-padded parts of short clips end up at the floor.
"""

import torch
from torch import nn

SAMPLE_RATE = 2000


class LogSpectrogram(nn.Module):
    def __init__(self, win_length: int = 256, hop_length: int = 16, n_fft: int = 512,
                 f_min: float = 40.0, f_max: float = 500.0):
        super().__init__()
        self.win_length, self.hop_length, self.n_fft = win_length, hop_length, n_fft
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        hz_per_bin = SAMPLE_RATE / n_fft
        self.first_bin = int(round(f_min / hz_per_bin))
        self.last_bin = int(round(f_max / hz_per_bin))  # exclusive

    @property
    def n_bins(self) -> int:
        return self.last_bin - self.first_bin

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """[batch, samples] -> [batch, 1, frequency bins, frames]"""
        stft = torch.stft(waveform, self.n_fft, self.hop_length, self.win_length, self.window,
                          center=True, pad_mode="constant", return_complex=True)
        power = stft[:, self.first_bin : self.last_bin].abs().square()
        db = 10 * torch.log10(power + 1e-6)
        db = db - db.median(dim=-1, keepdim=True).values
        return (db.clamp(-10.0, 30.0) / 10.0).unsqueeze(1)


def spec_augment(spec: torch.Tensor, max_freq_bins: int = 8, max_frames: int = 25) -> torch.Tensor:
    """Blank one frequency band and one time band per example (training only).

    The masks are small because the call is small: at most ~31 Hz and 0.2 s.
    Masked cells are set to 0, the per-row median level.
    """
    spec = spec.clone()
    batch, _, n_freq, n_time = spec.shape
    for i in range(batch):
        width = int(torch.randint(0, max_freq_bins + 1, ()))
        start = int(torch.randint(0, n_freq - width + 1, ()))
        spec[i, :, start : start + width, :] = 0.0
        width = int(torch.randint(0, max_frames + 1, ()))
        start = int(torch.randint(0, n_time - width + 1, ()))
        spec[i, :, :, start : start + width] = 0.0
    return spec
