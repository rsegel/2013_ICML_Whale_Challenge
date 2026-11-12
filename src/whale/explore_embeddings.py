"""Look at the structure of Perch embeddings next to the raw spectrograms.

Uses only the design fold's training days (28 and 29 March). Exploration is
part of model development, so it must not see the days held out for validation
or testing, nor the Kaggle days.

Takes the same random sample of clips from each training day, represents each
clip two ways -

* the 1,536-number Perch embedding, and
* the log spectrogram that the CNN baseline sees, flattened (29,618 numbers),

- and projects both to two dimensions with PCA and with UMAP, coloured by day,
by label and by clip length. Two numbers summarise what the pictures show: how
well a linear classifier can tell which day a clip came from (lower is better:
less day-specific nuisance), and how well it can tell an upcall (higher is
better).

Usage::

    python -m whale.explore_embeddings
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from whale import paths
from whale.data.splits import DESIGN_FOLD, FOLDS, load_labelled
from whale.data.preprocess import centre_pad, clean_clip, fade_edges
from whale.embed import embedding_path
from whale.features.perch_input import PerchInputConfig
from whale.features.spectrogram import LogSpectrogram

PER_DAY = 3000
SEED = 0
N_COMPONENTS = 50  # PCA dimensions fed to UMAP and to the linear classifiers
INPUT = PerchInputConfig(speed=4, pad="zero")
FIGURES = {kind: paths.REPO_ROOT / "reports" / "figures" / f"embedding_structure_{kind}.png" for kind in ("pca", "umap")}
RESULTS = paths.REPO_ROOT / "reports" / "embedding_structure.json"

# Okabe-Ito colour-blind-safe palette: labelled days first, Kaggle days after.
DAY_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#000000"]
LABEL_COLOURS = {0: "#7a9cc6", 1: "#D55E00"}
LABEL_NAMES = {0: "no upcall", 1: "upcall"}


def sample_clips() -> pd.DataFrame:
    """``PER_DAY`` random clips from each training day of the design fold."""
    manifest = load_labelled(DESIGN_FOLD)
    manifest["row"] = np.arange(len(manifest))
    train = manifest[manifest["split"] == "train"].assign(subset="train")
    assert set(train["date"]) == set(FOLDS[DESIGN_FOLD]["train"])
    return train.groupby("date", group_keys=False).sample(PER_DAY, random_state=SEED).reset_index(drop=True)


def load_representations(sample: pd.DataFrame) -> dict[str, np.ndarray]:
    embeddings, spectrograms = [], []
    to_spectrogram = LogSpectrogram()
    for subset, part in sample.groupby("subset", sort=False):
        rows = part["row"].to_numpy()
        embeddings.append(np.load(embedding_path(subset, INPUT), mmap_mode="r")[np.sort(rows)][np.argsort(np.argsort(rows))])
        waveforms = np.load(paths.waveform_path(subset), mmap_mode="r")
        clips = np.stack([centre_pad(fade_edges(clean_clip(waveforms[r], n)), paths.CLIP_SAMPLES)
                          for r, n in zip(rows, part["n_samples"])])
        with torch.no_grad():
            spectrograms.append(to_spectrogram(torch.from_numpy(clips)).flatten(1).numpy())
    return {"Perch embedding (1,536 numbers)": np.concatenate(embeddings),
            "Log spectrogram (29,618 numbers)": np.concatenate(spectrograms)}


def linear_separability(reduced: np.ndarray, sample: pd.DataFrame) -> dict[str, float]:
    """Cross-validated accuracy of a linear classifier on the PCA-reduced data.

    Folds keep each 15-minute chunk together, so the classifier cannot use the
    fact that neighbouring clips share background noise.
    """
    features = StandardScaler().fit_transform(reduced)
    groups = (sample["subset"] + sample["chunk_id"].astype(str)).to_numpy()
    day = cross_val_predict(LogisticRegression(max_iter=2000), features, sample["date"], groups=groups,
                            cv=GroupKFold(5))
    labelled = np.ones(len(sample), dtype=bool)
    upcall = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"), features[labelled],
                               sample["label"][labelled], groups=groups[labelled], cv=GroupKFold(5))
    truth = sample["label"][labelled].to_numpy()
    recalls = [(upcall[truth == c] == c).mean() for c in (0, 1)]
    return {"day_accuracy": float((day == sample["date"]).mean()), "day_chance": 1 / sample["date"].nunique(),
            "upcall_balanced_accuracy": float(np.mean(recalls))}


def scatter(axis, points: np.ndarray, colours: np.ndarray, order: np.ndarray):
    numeric = colours.dtype.kind == "f"
    drawn = axis.scatter(points[order, 0], points[order, 1], c=colours[order], s=3, alpha=0.55, linewidths=0,
                         rasterized=True, **({"cmap": "viridis", "vmin": 0, "vmax": 2} if numeric else {}))
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines[["top", "right", "left", "bottom"]].set_color("0.8")
    return drawn


def main() -> None:
    sample = sample_clips()
    representations = load_representations(sample)
    days = sorted(sample["date"].unique())
    day_colour = dict(zip(days, DAY_COLOURS))
    by_day = sample["date"].map(day_colour).to_numpy()
    by_label = sample["label"].map(LABEL_COLOURS).to_numpy()
    order = np.random.default_rng(SEED).permutation(len(sample))  # so no class is drawn on top

    by_length = (sample["n_samples"] / paths.SAMPLE_RATE).to_numpy(dtype=float)
    colourings = [(by_day, "by day"), (by_label, "by label"), (by_length, "by clip length")]

    results, points = {}, {"pca": {}, "umap": {}}
    for name, values in representations.items():
        standardised = StandardScaler().fit_transform(values)
        pca = PCA(N_COMPONENTS, random_state=SEED).fit(standardised)
        components = pca.transform(standardised)
        points["pca"][name] = components[:, :2]
        points["umap"][name] = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=SEED).fit_transform(components)
        short = by_length < 2
        results[name] = {**linear_separability(components, sample),
                         "variance_in_first_2_components": float(pca.explained_variance_ratio_[:2].sum()),
                         f"variance_in_first_{N_COMPONENTS}_components": float(pca.explained_variance_ratio_.sum()),
                         "correlation_of_clip_length_with_first_component":
                             float(np.corrcoef(by_length, components[:, 0])[0, 1]),
                         "correlation_within_short_clips":
                             float(np.corrcoef(by_length[short], components[short, 0])[0, 1])}
        print(name, json.dumps(results[name]), flush=True)

    def handles(mapping):
        return [plt.Line2D([], [], marker="o", ls="", color=c, label=label, ms=7) for label, c in mapping.items()]

    day_names = {f"{d[6:]} March": c for d, c in day_colour.items()}
    for kind, title in (("pca", "PCA: first two principal components"),
                        ("umap", f"UMAP of the first {N_COMPONENTS} principal components")):
        fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.4))
        for row, name in enumerate(representations):
            for axis, (colours, label) in zip(axes[row], colourings):
                drawn = scatter(axis, points[kind][name], colours, order)
                axis.set_title(label.capitalize(), fontsize=11)
            fig.colorbar(drawn, ax=axes[row, 2], fraction=0.046, pad=0.02, label="Clip length (seconds)")
            axes[row, 0].set_ylabel(name, fontsize=11)
        fig.legend(handles=handles(day_names), loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=4, frameon=False,
                   title="Day", title_fontsize=10)
        fig.legend(handles=handles({LABEL_NAMES[k]: c for k, c in LABEL_COLOURS.items()}), loc="lower center",
                   bbox_to_anchor=(0.62, 0.0), ncol=1, frameon=False, title="Label", title_fontsize=10)
        fig.suptitle(f"{title}. Training days of the design fold only, {PER_DAY:,} random clips per day; "
                     "each dot is one clip.", fontsize=11)
        fig.tight_layout(rect=(0, 0.09, 1, 0.97))
        fig.savefig(FIGURES[kind], dpi=130)
        plt.close(fig)
    RESULTS.write_text(json.dumps({"clips_per_day": PER_DAY, "days": days, **results}, indent=2))
    print("wrote", *FIGURES.values(), "and", RESULTS)


if __name__ == "__main__":
    main()
