"""Fit a logistic-regression probe on frozen Perch embeddings for one fold.

Mirrors ``whale.train``: fits on the fold's training days, picks the
regularisation strength by validation average precision, then writes that
probe's scores for the validation and test days in the same format. The test
day is only scored, never used for any decision.

Usage::

    python -m whale.probe --fold 0
"""

import argparse
import time

import numpy as np
import pandas as pd
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from whale import paths
from whale.data.splits import FOLDS, load_labelled
from whale.embed import embedding_path
from whale.features.perch_input import PerchInputConfig
from whale.metrics import ranking_metrics
from whale.train import PROJECT, subsample_training_rows, wandb_mode

NAME = "perch_logreg"
C_GRID = (0.0001, 0.001, 0.01, 0.1)
INPUT = PerchInputConfig(speed=4, pad="zero")  # chosen in select_perch_input


def run(fold: int, seed: int = 0, train_fraction: float = 1.0, limit: int | None = None):
    manifest = load_labelled(fold)
    embeddings = np.load(embedding_path("train", INPUT), mmap_mode="r")
    assert len(embeddings) == len(manifest), "embedding file does not match the manifest"
    rows = {split: manifest.index[manifest["split"] == split].to_numpy() for split in ("train", "val", "test")}
    rows["train"] = subsample_training_rows(manifest, rows["train"], train_fraction, seed)
    if limit:  # smoke tests: an evenly spaced subset, so both classes are present
        rows = {split: r[np.linspace(0, len(r) - 1, limit).astype(int)] for split, r in rows.items()}
    features = {split: np.asarray(embeddings[r]) for split, r in rows.items()}
    labels = {split: manifest.loc[r, "label"].to_numpy() for split, r in rows.items()}

    run_name = f"{NAME}_fold{fold}_seed{seed}"
    if train_fraction < 1:
        run_name += f"_frac{train_fraction:g}"
    wandb.init(project=PROJECT, name=run_name, group=NAME, job_type="train", mode=wandb_mode(),
               config={"name": NAME, "fold": fold, "fold_days": FOLDS[fold], "seed": seed,
                       "input": INPUT.name, "C_grid": C_GRID, "train_fraction": train_fraction,
                       "n_train": len(rows["train"]), "n_val": len(rows["val"]),
                       "trainable_parameters": features["train"].shape[1] + 1})

    started = time.perf_counter()
    best = None
    for c in C_GRID:
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
        probe.fit(features["train"], labels["train"])
        val = ranking_metrics(labels["val"], probe.predict_proba(features["val"])[:, 1])
        wandb.log({"C": c, **{f"val/{k}": v for k, v in val.items()}})
        print(f"C {c:<6}  val AUC {val['roc_auc']:.4f}  val AP {val['average_precision']:.4f}  "
              f"{(time.perf_counter() - started) / 60:.1f} min", flush=True)
        if best is None or val["average_precision"] > best["val"]["average_precision"]:
            best = {"C": c, "val": val, "probe": probe}

    frames = [pd.DataFrame({
        "clip_index": manifest.loc[rows[split], "clip_index"].to_numpy(), "row": rows[split],
        "split": split, "label": labels[split].astype(int),
        "score": best["probe"].predict_proba(features[split])[:, 1]}) for split in ("val", "test")]
    predictions = pd.concat(frames).assign(model=NAME, fold=fold, seed=seed, train_fraction=train_fraction)
    out = paths.CACHE_DIR / "predictions" / f"{run_name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(out, index=False)

    wandb.summary.update({"best_C": best["C"], "best_val_average_precision": best["val"]["average_precision"],
                          "best_val_roc_auc": best["val"]["roc_auc"],
                          "train_minutes": (time.perf_counter() - started) / 60})
    wandb.finish()
    print(f"best C {best['C']}  predictions -> {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fold", type=int, default=0, choices=list(FOLDS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--limit", type=int, help="use only N clips of each split (smoke test)")
    args = parser.parse_args()
    run(args.fold, args.seed, args.train_fraction, args.limit)
