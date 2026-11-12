import numpy as np
import pandas as pd
import pytest

from whale import paths
from whale.evaluate import bootstrap, common_folds, error_overlap, fold_metrics


def synthetic_predictions(models=("good", "weak"), folds=(0, 1), n=600, seed=0) -> pd.DataFrame:
    """Two models scoring the same clips; "good" separates the classes better."""
    rng = np.random.default_rng(seed)
    frames, row = [], 0
    for fold in folds:
        for split in ("val", "test"):
            labels = (rng.random(n) < 0.2).astype(int)
            rows = np.arange(row, row + n)
            row += n
            for model in models:
                noise = 0.15 if model == "good" else 0.45
                scores = np.clip(0.2 + 0.6 * labels + noise * rng.standard_normal(n), 0, 1)
                frames.append(pd.DataFrame({
                    "row": rows, "split": split, "label": labels, "score": scores, "model": model,
                    "fold": fold, "chunk_id": rows // 20, "n_samples": paths.CLIP_SAMPLES}))
    return pd.concat(frames, ignore_index=True)


def test_common_folds_keeps_only_shared_folds():
    pred = synthetic_predictions()
    pred = pred[~((pred["model"] == "weak") & (pred["fold"] == 1))]
    assert common_folds(pred) == [0]


def test_fold_metrics_uses_the_validation_threshold():
    pred = synthetic_predictions(models=("good",), folds=(0,))
    val, test = pred[pred["split"] == "val"], pred[pred["split"] == "test"]
    out = fold_metrics(val, test)
    assert out["clips"] == 600 and out["roc_auc"] > 0.95
    # The threshold is the validation day's, so test recall is near 0.90, not exactly it.
    assert out["recall_at_90"] == pytest.approx(0.90, abs=0.08)
    assert out["threshold_at_95"] <= out["threshold_at_90"]
    assert out["fp_per_1000_at_95"] >= out["fp_per_1000_at_90"]


def test_bootstrap_separates_models_and_is_paired():
    pred = synthetic_predictions()
    out = bootstrap(pred, "test", [0, 1], n_boot=100)
    low, high = out["good - weak"]["roc_auc"]
    assert low > 0  # the better model is distinguishable
    # A model compared with a copy of itself differs by exactly zero in every resample.
    twin = pd.concat([pred[pred["model"] == "good"], pred[pred["model"] == "good"].assign(model="twin")])
    assert bootstrap(twin, "test", [0, 1], n_boot=20)["good - twin"]["roc_auc"] == [0.0, 0.0]


def test_error_overlap_counts_each_wrong_clip_once():
    pred = synthetic_predictions()
    overlap = error_overlap(pred, "test", [0, 1])
    assert set(overlap.index) == {"false alarm", "missed call"}
    assert (overlap >= 0).all().all()
    positives = pred[(pred["split"] == "test") & (pred["model"] == "good")]["label"].sum()
    assert overlap.loc["missed call"].sum() <= positives
