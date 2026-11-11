import numpy as np
import pandas as pd
import pytest
import torch

from whale.data.dataset import ClipDataset
from whale.features.spectrogram import LogSpectrogram, spec_augment
from whale.metrics import operating_point, threshold_for_recall
from whale.models.resnet import SpectrogramResNet
from whale.train import warmup_cosine


def tone(freq: float, seconds: float = 2.0) -> torch.Tensor:
    t = torch.arange(int(2000 * seconds)) / 2000
    return torch.sin(2 * torch.pi * freq * t) + 0.01 * torch.randn(len(t))


def test_spectrogram_shape_and_band():
    spec = LogSpectrogram()
    out = spec(torch.randn(3, 4000))
    assert out.shape == (3, 1, 118, 251)
    assert spec.first_bin * 2000 / 512 == pytest.approx(40, abs=4)


def test_steady_tone_is_removed_but_a_burst_is_kept():
    """Per-row median subtraction flattens a constant band and keeps a transient."""
    spec = LogSpectrogram()
    steady = spec(tone(150.0)[None])[0, 0]
    tone_rows = slice(26, 31)  # (150 Hz - 40 Hz) / 3.9 Hz per bin = row 28
    # Edge frames are skipped: a tone that starts abruptly splatters there.
    assert steady[tone_rows, 20:-20].abs().max() < 0.5
    burst = torch.zeros(4000) + 0.01 * torch.randn(4000)
    burst[1600:2400] += tone(150.0, 0.4)
    assert spec(burst[None])[0, 0].max() > 2.0


def test_spec_augment_keeps_shape_and_only_zeroes():
    spec = torch.ones(4, 1, 118, 251)
    out = spec_augment(spec)
    assert out.shape == spec.shape
    assert set(out.unique().tolist()) <= {0.0, 1.0}
    assert spec.min() == 1.0  # input not modified


def make_dataset(max_shift: int):
    waveforms = np.zeros((2, 4000), dtype=np.int16)
    waveforms[0] = np.random.default_rng(0).normal(-700, 300, 4000)
    waveforms[1, :2000] = np.random.default_rng(1).normal(-700, 300, 2000)
    manifest = pd.DataFrame({"n_samples": [4000, 2000], "label": [1, 0]})
    return ClipDataset(manifest, waveforms, np.array([0, 1]), max_shift)


def test_dataset_centres_short_clips():
    waveform, label, row = make_dataset(0)[1]
    assert waveform.shape == (4000,) and label == 0 and row == 1
    assert waveform[:1000].abs().sum() == 0 and waveform[3000:].abs().sum() == 0
    assert waveform[1000:3000].std() == pytest.approx(1.0, abs=0.01)


def test_dataset_fades_clip_edges():
    waveform, _, _ = make_dataset(0)[1]  # real audio occupies samples 1000-2999
    assert waveform[1000] == 0 and abs(waveform[2999]) < 0.05 * waveform[1000:3000].abs().max()
    assert waveform[1000:1010].abs().mean() < 0.5 * waveform[1100:2900].abs().mean()


def test_dataset_shift_stays_in_range():
    dataset = make_dataset(500)
    for _ in range(20):
        waveform, _, _ = dataset[1]
        active = torch.nonzero(waveform).squeeze()
        assert 500 <= active.min() and active.max() < 3500


def test_frozen_model_trains_only_the_head():
    model = SpectrogramResNet(pretrained=False, freeze_backbone=True)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable == ["backbone.fc.weight", "backbone.fc.bias"]
    model.train()
    assert not model.backbone.bn1.training
    assert model(torch.randn(2, 4000)).shape == (2,)


def test_warmup_cosine_schedule():
    assert warmup_cosine(0, 10, 100) == pytest.approx(0.1)
    assert warmup_cosine(9, 10, 100) == pytest.approx(1.0)
    assert warmup_cosine(100, 10, 100) == pytest.approx(0.0, abs=1e-9)


def test_threshold_for_recall():
    labels = np.array([0] * 90 + [1] * 10)
    scores = np.concatenate([np.linspace(0, 0.5, 90), np.linspace(0.3, 1.0, 10)])
    threshold = threshold_for_recall(labels, scores, 0.9)
    point = operating_point(labels, scores, threshold)
    assert point["recall"] >= 0.9
    assert point["tp"] + point["fn"] == 10 and point["fp"] + point["tn"] == 90
