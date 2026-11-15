"""Score the unlabelled Kaggle clips (4-6 April) with one of the two reported models.

Cross-validation leaves four models per approach, one per fold. Each scores
every Kaggle clip and the four probabilities are averaged, so the submission
uses exactly the models that were evaluated: nothing is retrained and no new
choice is made. The Kaggle clips have no labels, so nothing is measured here.

* ``perch``: the probes are not stored, so each fold's logistic probe is refitted
  on its training days (seconds) with the regularisation strength picked on its
  validation day, as in ``whale.probe``. The refit is checked against the
  validation scores saved by the original run.
* ``resnet``: loads each fold's best checkpoint written by ``whale.train``.

Writes ``submissions/<model>.csv`` (``clip,probability``, in the order of
Kaggle's sample submission) and the per-fold scores to the cache.

Usage::

    python -m whale.embed test --speed 4 --pad zero --denoise whiten   # once, for perch
    python -m whale.predict perch
    python -m whale.predict resnet --checkpoints path/to/checkpoints
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from whale import paths
from whale.data.dataset import ClipDataset
from whale.data.splits import FOLDS, load_labelled
from whale.embed import embedding_path
from whale.features.perch_input import PerchInputConfig
from whale.metrics import ranking_metrics
from whale.models.resnet import SpectrogramResNet
from whale.probe import INPUT, MODELS
from whale.train import predict, sigmoid

PERCH_NAME, RESNET_NAME = "perch_logreg_whiten", "resnet18_finetune"
PERCH_INPUT = PerchInputConfig(INPUT.speed, INPUT.pad, INPUT.peak, "whiten")
SUBMISSIONS = paths.REPO_ROOT / "submissions"


def perch_fold_scores(fold: int, kaggle_embeddings: np.ndarray) -> np.ndarray:
    """Refit one fold's probe as ``whale.probe`` does and score the Kaggle clips."""
    _, grid, build = MODELS["logreg"]
    manifest = load_labelled(fold)
    embeddings = np.load(embedding_path("train", PERCH_INPUT), mmap_mode="r")
    assert len(embeddings) == len(manifest), "embedding file does not match the manifest"
    rows = {split: manifest.index[manifest["split"] == split].to_numpy() for split in ("train", "val")}
    features = {split: np.asarray(embeddings[r]) for split, r in rows.items()}
    labels = {split: manifest.loc[r, "label"].to_numpy() for split, r in rows.items()}

    best_ap, best_c, best_probe = -1.0, None, None
    for c in grid:
        probe = make_pipeline(StandardScaler(), build(c)).fit(features["train"], labels["train"])
        ap = ranking_metrics(labels["val"], probe.predict_proba(features["val"])[:, 1])["average_precision"]
        if ap > best_ap:
            best_ap, best_c, best_probe = ap, c, probe

    message = f"fold {fold}: C {best_c}, val AP {best_ap:.4f}"
    saved = paths.CACHE_DIR / "predictions" / f"{PERCH_NAME}_fold{fold}_seed0.parquet"
    if saved.exists():  # the refit must reproduce the evaluated model
        original = pd.read_parquet(saved).query("split == 'val'").set_index("row")["score"]
        gap = np.abs(best_probe.predict_proba(features["val"])[:, 1] - original.loc[rows["val"]].to_numpy()).max()
        assert gap < 1e-4, f"fold {fold}: refitted probe differs from the saved run (max gap {gap:.2g})"
        message += f", matches the saved validation scores (max gap {gap:.1e})"
    print(message, flush=True)
    return best_probe.predict_proba(kaggle_embeddings)[:, 1]


def resnet_fold_scores(fold: int, checkpoints: Path, manifest: pd.DataFrame, waveforms: np.ndarray,
                       rows: np.ndarray, device: torch.device) -> np.ndarray:
    """Score the Kaggle clips with one fold's best checkpoint."""
    file = checkpoints / f"{RESNET_NAME}_fold{fold}_seed0.pt"
    if not file.exists():
        raise FileNotFoundError(f"{file} is missing: copy the checkpoint written by whale.train for fold {fold}")
    model = SpectrogramResNet(pretrained=False, augment=False).to(device)
    model.load_state_dict(torch.load(file, map_location=device))
    loader = DataLoader(ClipDataset(manifest, waveforms, rows), 256)  # centred, no augmentation
    scored_rows, _, logits = predict(model, loader)
    assert (scored_rows == rows).all()
    print(f"fold {fold}: scored {len(rows)} clips with {file.name}", flush=True)
    return sigmoid(logits)


def main(model: str, checkpoints: Path, limit: int | None = None) -> Path:
    manifest = pd.read_parquet(paths.manifest_path("test"))
    rows = np.arange(len(manifest) if limit is None else limit)
    name = PERCH_NAME if model == "perch" else RESNET_NAME

    if model == "perch":
        embeddings = np.load(embedding_path("test", PERCH_INPUT), mmap_mode="r")
        assert len(embeddings) == len(manifest), "embedding file does not match the manifest"
        kaggle = np.asarray(embeddings[rows])
        per_fold = {fold: perch_fold_scores(fold, kaggle) for fold in FOLDS}
    else:
        waveforms = np.load(paths.waveform_path("test"), mmap_mode="r")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        per_fold = {fold: resnet_fold_scores(fold, checkpoints, manifest, waveforms, rows, device) for fold in FOLDS}

    scores = pd.DataFrame({f"fold{fold}": s for fold, s in per_fold.items()})
    scores.insert(0, "clip", manifest["filename"].to_numpy()[rows])
    scores["probability"] = scores[[f"fold{fold}" for fold in FOLDS]].mean(axis=1)
    assert scores["probability"].between(0, 1).all() and not scores.isna().any().any()

    if limit:  # keep smoke tests away from the real files
        name += f"_limit{limit}"
    else:
        sample = paths.DATA_DIR / "sampleSubmission.csv"
        if sample.exists():  # Kaggle's row order
            order = pd.read_csv(sample)["clip"]
            assert set(order) == set(scores["clip"]) and len(order) == len(scores), "clips differ from the sample"
            scores = scores.set_index("clip").loc[order].reset_index()

    per_fold_file = paths.CACHE_DIR / "predictions" / "kaggle" / f"{name}.parquet"
    per_fold_file.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(per_fold_file, index=False)
    SUBMISSIONS.mkdir(exist_ok=True)
    out = SUBMISSIONS / f"{name}.csv"
    scores[["clip", "probability"]].to_csv(out, index=False)

    print(f"\n{len(scores)} clips; share scored >= 0.5: {(scores['probability'] >= 0.5).mean():.3f}; "
          f"mean probability {scores['probability'].mean():.3f}")
    print("agreement between folds (Spearman):\n" +
          scores[[f"fold{fold}" for fold in FOLDS]].corr(method="spearman").round(3).to_string())
    print(f"submission -> {out}\nper-fold scores -> {per_fold_file}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model", choices=["perch", "resnet"])
    parser.add_argument("--checkpoints", type=Path, default=paths.CACHE_DIR / "checkpoints",
                        help="folder holding resnet18_finetune_fold{0..3}_seed0.pt")
    parser.add_argument("--limit", type=int, help="score only the first N clips (smoke test)")
    args = parser.parse_args()
    main(args.model, args.checkpoints, args.limit)
