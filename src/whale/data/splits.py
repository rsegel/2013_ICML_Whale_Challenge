"""Cross-validation folds over the four labelled recording days.

Clips are detector output from continuous recordings: neighbours overlap in
time, share background noise, and their labels are correlated (on the design
fold's training days, a clip that follows a positive is positive 23-26% of the
time against an 8-11% base rate). A random split would therefore leak, so whole
days are held out. That
also mirrors the real task: Kaggle's unlabelled test set is from later days.

Each fold tests on one day, validates (early stopping, threshold) on the day
before it, and trains on the other two. Every day is the test day exactly
once, so every clip gets one out-of-fold test prediction.

Fold 0 is the **design fold**: input-format and hyperparameter choices are
made on its validation day only and then frozen. Its test day (31 March) is
therefore the one day no design decision has ever seen.

Usage::

    python -m whale.data.splits
"""

import pandas as pd

from whale import paths

DAYS = ("20090328", "20090329", "20090330", "20090331")

# fold -> {split name: days}
FOLDS = {
    0: {"train": ("20090328", "20090329"), "val": ("20090330",), "test": ("20090331",)},
    1: {"train": ("20090329", "20090330"), "val": ("20090331",), "test": ("20090328",)},
    2: {"train": ("20090330", "20090331"), "val": ("20090328",), "test": ("20090329",)},
    3: {"train": ("20090331", "20090328"), "val": ("20090329",), "test": ("20090330",)},
}
DESIGN_FOLD = 0


def assign_splits(manifest: pd.DataFrame, fold: int = DESIGN_FOLD) -> pd.Series:
    """Return the split name of every row for one fold; fail on an unknown day."""
    day_to_split = {day: split for split, days in FOLDS[fold].items() for day in days}
    unknown = set(manifest["date"]) - set(day_to_split)
    if unknown:
        raise ValueError(f"Days with no split assignment: {sorted(unknown)}")
    return manifest["date"].map(day_to_split).rename("split")


def load_labelled(fold: int = DESIGN_FOLD) -> pd.DataFrame:
    """Labelled manifest with a ``split`` column for ``fold``. Row i is waveform i."""
    manifest = pd.read_parquet(paths.manifest_path("train"))
    manifest["split"] = assign_splits(manifest, fold)
    return manifest


def summarise(manifest: pd.DataFrame) -> pd.DataFrame:
    table = manifest.groupby("split")["label"].agg(clips="size", positive="sum")
    table["negative"] = table["clips"] - table["positive"]
    table["pct_positive"] = (100 * table["positive"] / table["clips"]).round(1)
    table["pct_of_clips"] = (100 * table["clips"] / len(manifest)).round(1)
    table["chunks"] = manifest.groupby("split")["chunk_id"].nunique()
    short = manifest["n_samples"] < paths.CLIP_SAMPLES
    table["pct_short"] = (100 * short.groupby(manifest["split"]).mean()).round(1)
    return table.loc[["train", "val", "test"]]


if __name__ == "__main__":
    for fold, days in FOLDS.items():
        print(f"\nfold {fold}: " + ", ".join(f"{k}={'+'.join(d[-2:] for d in v)} Mar" for k, v in days.items()))
        print(summarise(load_labelled(fold)).to_string())
