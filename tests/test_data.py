import numpy as np
import pytest

from whale.data.preprocess import centre_pad, clean_clip
from whale.features.perch_input import PerchInputConfig, make_perch_input
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
