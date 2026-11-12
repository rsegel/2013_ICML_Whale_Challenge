import numpy as np
import pytest

from whale.data.preprocess import centre_pad, clean_clip
from whale.features.perch_input import PerchInputConfig, denoise_clip, make_perch_input
from whale.data.manifest import parse_filename


def test_parse_train_filename():
    row = parse_filename("20090328_001500_189s6ms_TRAIN16_1.aif")
    assert row["date"] == "20090328"
    assert row["chunk_id"] == "20090328_001500"
    assert row["offset_s"] == pytest.approx(189.6)
    assert row["clip_index"] == 16
    assert row["label"] == 1


def test_parse_test_filename_has_no_label():
    row = parse_filename("20090404_000000_012s0ms_Test0.aif")
    assert row["label"] == -1
    assert row["clip_index"] == 0


def test_parse_rejects_unknown_names():
    with pytest.raises(ValueError):
        parse_filename("notes.txt")


def test_centre_pad_short_clip():
    out = centre_pad(np.full(1000, 7, dtype=np.int16), 4000)
    assert out.shape == (4000,)
    assert (out[1500:2500] == 7).all()
    assert out[:1500].sum() == 0 and out[2500:].sum() == 0


def test_centre_pad_full_clip_unchanged():
    clip = np.arange(4000, dtype=np.int16)
    assert (centre_pad(clip, 4000) == clip).all()


def test_clean_clip_removes_offset_and_level():
    rng = np.random.default_rng(0)
    cached = np.zeros(4000, dtype=np.int16)
    cached[:3000] = (rng.normal(-730, 300, 3000)).astype(np.int16)
    clip = clean_clip(cached, 3000)
    assert clip.shape == (3000,) and clip.dtype == np.float32
    assert abs(clip.mean()) < 1e-4 and abs(clip.std() - 1) < 1e-4


@pytest.mark.parametrize("speed,pad", [(1, "zero"), (4, "zero"), (8, "tile")])
def test_perch_input_shape_and_peak(speed, pad):
    clip = np.random.default_rng(0).normal(size=4000).astype(np.float32)
    out = make_perch_input(clip, PerchInputConfig(speed, pad))
    assert out.shape == (160_000,) and out.dtype == np.float32
    assert np.abs(out).max() == pytest.approx(0.25, rel=1e-4)
    if pad == "zero":  # audio occupies 2/speed seconds in the middle
        active = np.flatnonzero(out)
        assert active[-1] - active[0] < 32_000 * 2 / speed


@pytest.mark.parametrize("method", ["whiten", "subtract"])
def test_denoise_removes_steady_tone_and_keeps_sweep(method):
    """A constant 60 Hz tone (like the tonal noise bands) is suppressed relative
    to a short 100->200 Hz sweep (like an upcall)."""
    t = np.arange(4000) / 2000
    tone = 5 * np.sin(2 * np.pi * 60 * t)
    burst = (t > 0.6) & (t < 1.4)
    sweep = np.where(burst, np.sin(2 * np.pi * (100 * (t - 0.6) + 62.5 * (t - 0.6) ** 2)), 0.0)
    out = denoise_clip(tone + sweep, method)
    assert out.shape == t.shape

    def band_power(x, f_lo, f_hi):
        spectrum = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1 / 2000)
        return spectrum[(f >= f_lo) & (f < f_hi)].sum()

    before = band_power(tone + sweep, 55, 65) / band_power(tone + sweep, 110, 190)
    after = band_power(out, 55, 65) / band_power(out, 110, 190)
    assert after < before / 10


def test_denoise_none_is_identity_and_names_unchanged():
    clip = np.random.default_rng(0).normal(size=1000)
    assert denoise_clip(clip, "none") is clip
    assert PerchInputConfig(4, "zero").name == "speed4_zero_peak0.25"
    assert PerchInputConfig(4, "zero", denoise="whiten").name == "speed4_zero_peak0.25_whiten"
