"""Choose how to present 2 kHz clips to Perch, on the design fold only.

For each input variant: embed the same fixed subset of training and validation
clips, fit a logistic-regression probe on the training part, score the
validation part. The same clips are used for every variant, so differences are
paired. The subset is enriched in positives to tighten the ROC-AUC estimate
(ROC-AUC does not depend on class balance; average precision does, so the AP
here is only comparable between variants, not with later results).

The fold's test day is never read.

Usage::

    python -m whale.select_perch_input
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from whale import paths
from whale.data.splits import DESIGN_FOLD, load_labelled
from whale.embed import embed_rows
from whale.features.perch_input import PerchInputConfig
from whale.models.perch_onnx import PerchEmbedder

VARIANTS = [
    PerchInputConfig(1, "zero"),
    PerchInputConfig(4, "zero"),
    PerchInputConfig(4, "tile"),
    PerchInputConfig(16, "zero"),
    PerchInputConfig(2, "zero"),
    PerchInputConfig(8, "zero"),
    PerchInputConfig(1, "tile"),
    PerchInputConfig(4, "zero", peak=1.0),
    # waveform-domain versions of the CNN's per-frequency median subtraction
    PerchInputConfig(4, "zero", denoise="whiten"),
    PerchInputConfig(4, "zero", denoise="subtract"),
    # complete the speed x padding grid
    PerchInputConfig(2, "tile"),
    PerchInputConfig(8, "tile"),
    PerchInputConfig(16, "tile"),
    PerchInputConfig(16, "tile", denoise="whiten"),
]
SUBSET = {"train": (1000, 2000), "val": (600, 1400)}  # (positives, negatives)
C_GRID = (0.001, 0.01, 0.1)
SEED = 0

CACHE = paths.CACHE_DIR / "perch_select"
RESULTS = paths.REPO_ROOT / "reports" / "perch_input_selection.json"


def pick_rows(manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    rows = {}
    for split, (n_pos, n_neg) in SUBSET.items():
        part = manifest[manifest["split"] == split]
        pos = rng.choice(part.index[part["label"] == 1], n_pos, replace=False)
        neg = rng.choice(part.index[part["label"] == 0], n_neg, replace=False)
        rows[split] = np.sort(np.concatenate([pos, neg]))
    return rows


def chunk_bootstrap_auc_difference(labels, scores_a, scores_b, chunks, n_boot=1000) -> tuple[float, float]:
    """95% interval of AUC(a) - AUC(b), resampling 15-minute chunks."""
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(chunks == c) for c in np.unique(chunks)]
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        if labels[idx].min() == labels[idx].max():
            continue
        diffs.append(roc_auc_score(labels[idx], scores_a[idx]) - roc_auc_score(labels[idx], scores_b[idx]))
    return tuple(np.percentile(diffs, [2.5, 97.5]))


def main() -> None:
    manifest = load_labelled(DESIGN_FOLD)
    waveforms = np.load(paths.waveform_path("train"), mmap_mode="r")
    lengths = manifest["n_samples"].to_numpy()
    rows = pick_rows(manifest)
    labels = {split: manifest.loc[r, "label"].to_numpy() for split, r in rows.items()}
    val_chunks = manifest.loc[rows["val"], "chunk_id"].to_numpy()
    val_full = (manifest.loc[rows["val"], "n_samples"] == paths.CLIP_SAMPLES).to_numpy()

    CACHE.mkdir(parents=True, exist_ok=True)
    embedder = None
    results, val_scores = [], {}
    for config in VARIANTS:
        features = {}
        for split in ("train", "val"):
            file = CACHE / f"{split}_{config.name}.npy"
            if not file.exists():
                embedder = embedder or PerchEmbedder()
                np.save(file, embed_rows(embedder, waveforms, lengths, rows[split], config,
                                         progress=f"{config.name} {split}"))
            features[split] = np.load(file)

        best = None
        for c in C_GRID:
            probe = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
            probe.fit(features["train"], labels["train"])
            scores = probe.predict_proba(features["val"])[:, 1]
            auc = roc_auc_score(labels["val"], scores)
            if best is None or auc > best["val_auc"]:
                best = {"C": c, "val_auc": auc, "scores": scores}
        scores = best.pop("scores")
        val_scores[config.name] = scores
        results.append({
            "variant": config.name, "speed": config.speed, "pad": config.pad, "peak": config.peak, **best,
            "val_ap_enriched": average_precision_score(labels["val"], scores),
            "val_auc_full_length_only": roc_auc_score(labels["val"][val_full], scores[val_full]),
        })
        print(json.dumps(results[-1]), flush=True)
        RESULTS.write_text(json.dumps(results, indent=2))

    top = max(results, key=lambda r: r["val_auc"])
    for r in results:
        low, high = chunk_bootstrap_auc_difference(
            labels["val"], val_scores[top["variant"]], val_scores[r["variant"]], val_chunks)
        r["auc_gap_to_best_ci95"] = [round(low, 4), round(high, 4)]
    RESULTS.write_text(json.dumps(results, indent=2))
    table = pd.DataFrame(results).sort_values("val_auc", ascending=False)
    print("\n" + table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
