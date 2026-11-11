import pandas as pd
import pytest

from whale import paths
from whale.data.splits import DAYS, FOLDS, assign_splits, load_labelled


def test_unknown_day_is_rejected():
    with pytest.raises(ValueError):
        assign_splits(pd.DataFrame({"date": ["20090328", "20090401"]}))


def test_folds_partition_the_days():
    for days in FOLDS.values():
        used = [d for group in days.values() for d in group]
        assert sorted(used) == sorted(DAYS)  # every day used once, no overlap


def test_every_day_is_test_once_and_val_once():
    assert sorted(d for f in FOLDS.values() for d in f["test"]) == sorted(DAYS)
    assert sorted(d for f in FOLDS.values() for d in f["val"]) == sorted(DAYS)


@pytest.mark.parametrize("fold", list(FOLDS))
def test_no_day_or_chunk_spans_two_splits(fold):
    if not paths.manifest_path("train").exists():
        pytest.skip("decoded cache not built")
    labelled = load_labelled(fold)
    assert labelled["split"].isin(["train", "val", "test"]).all()
    assert labelled["clip_index"].is_unique
    assert (labelled.groupby("date")["split"].nunique() == 1).all()
    assert (labelled.groupby("chunk_id")["split"].nunique() == 1).all()


def test_consecutive_days_do_not_share_audio():
    """The last clip of a day ends before the first clip of the next begins."""
    if not paths.manifest_path("train").exists():
        pytest.skip("decoded cache not built")
    labelled = load_labelled()
    start = pd.to_datetime(labelled["date"]) + pd.to_timedelta(labelled["offset_s"], unit="s")
    first, last = start.groupby(labelled["date"]).min(), start.groupby(labelled["date"]).max()
    for earlier, later in zip(DAYS, DAYS[1:]):
        assert (first[later] - last[earlier]).total_seconds() > 2.0
