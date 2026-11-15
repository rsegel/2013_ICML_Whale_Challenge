"""Fold-0 test-day mistakes on the report's own PCA and UMAP maps.

The maps are not refitted: ``explore_embeddings`` saves its fitted scaler, PCA and
UMAP (``FITTED_MAPS``), and this script projects into exactly those, so the
section-3 and section-4 maps are the same maps. (Refitting is not an option: on
15 Nov the same data, code and seed gave a different UMAP layout than on 12 Nov.)
Run ``python -m whale.explore_embeddings`` first.

Rows, as in the section-3 figure:

* the Perch embedding of the plain input. Whitened Perch reads the embedding of
  the whitened input, so this is the shared map, not that model's own input;
* the log spectrogram the CNN sees (``LogSpectrogram``).

All clips of the fold-0 test day (31 March) are projected in with ``transform``.
Background dots are the test clips coloured by their label; the mistakes at each
model's validation-day 90%-recall threshold are drawn on top, split into only
whitened Perch wrong, only ResNet wrong, and both wrong.

    python -m whale.figure_test_error_maps
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
import torch

from whale import paths
from whale.data.preprocess import centre_pad, clean_clip, fade_edges
from whale.data.splits import load_labelled
from whale.embed import embedding_path
from whale.explore_embeddings import (FITTED_MAPS, INPUT, LABEL_COLOURS, N_COMPONENTS, SEED,
                                      sample_clips)
from whale.features.spectrogram import LogSpectrogram
from whale.figure_test_error_spectrograms import wrong_table

FIGURES = {kind: paths.REPO_ROOT / "reports" / "figures" / f"test_errors_{kind}_fold0.png" for kind in ("pca", "umap")}
ERROR_STYLE = {0: dict(marker="^", color="#0b2a6f", label="false alarm (no upcall, flagged)"),
               1: dict(marker="s", color="#000000", label="missed call (upcall, not flagged)")}


def spectrograms(rows: np.ndarray, n_samples: np.ndarray, batch: int = 1000) -> np.ndarray:
    waveforms = np.load(paths.waveform_path("train"), mmap_mode="r")
    layer = LogSpectrogram()
    out = []
    for start in range(0, len(rows), batch):
        clips = np.stack([centre_pad(fade_edges(clean_clip(waveforms[r], n)), paths.CLIP_SAMPLES)
                          for r, n in zip(rows[start:start + batch], n_samples[start:start + batch])])
        with torch.no_grad():
            out.append(layer(torch.from_numpy(clips)).flatten(1).numpy().astype(np.float32))
    return np.concatenate(out)


def project(fitted: dict, test: np.ndarray, batch: int = 2000) -> dict[str, np.ndarray]:
    pc = np.concatenate([fitted["pca"].transform(fitted["scaler"].transform(test[i:i + batch]))
                         for i in range(0, len(test), batch)])
    return {"pca": pc[:, :2], "umap": fitted["umap"].transform(pc)}


def main() -> None:
    saved = joblib.load(FITTED_MAPS)
    manifest = load_labelled(0)
    table = wrong_table()  # index = row of the test day, one column per model
    rows = table.index.to_numpy()
    assert set(manifest.loc[rows, "date"]) == {"20090331"}
    assert np.array_equal(saved["rows"], sample_clips()["row"].to_numpy()), "maps were fitted on other clips"

    test_rows = np.sort(rows)
    plain = np.load(embedding_path("train", INPUT), mmap_mode="r")
    representations = {
        "Perch embedding, plain input\n(the section-3 map)":
            (saved["maps"]["Perch embedding (1,536 numbers)"], np.asarray(plain[test_rows])),
        "Log spectrogram\n(what ResNet-18 reads; the section-3 map)":
            (saved["maps"]["Log spectrogram (29,618 numbers)"],
             spectrograms(test_rows, manifest.loc[test_rows, "n_samples"].to_numpy())),
    }
    table = table.loc[np.sort(rows)]
    labels = table["label"].to_numpy()
    p, r = table["perch_logreg_whiten"].to_numpy(), table["resnet18_finetune"].to_numpy()
    groups = {"only whitened Perch wrong": p & ~r, "only ResNet wrong": ~p & r, "both wrong": p & r}

    points = {"pca": {}, "umap": {}}
    for name, (fitted, test) in representations.items():
        projected = project(fitted, test)
        for kind in points:
            points[kind][name] = projected[kind]
        print("projected", name.splitlines()[0], flush=True)

    order = np.random.default_rng(SEED).permutation(len(labels))
    for kind, title in (("pca", "PCA: first two principal components"),
                        ("umap", f"UMAP of the first {N_COMPONENTS} principal components")):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for i, name in enumerate(representations):
            xy = points[kind][name]
            for ax, (group, mask) in zip(axes[i], groups.items()):
                ax.scatter(xy[order, 0], xy[order, 1], c=[LABEL_COLOURS[l] for l in labels[order]], s=3,
                           alpha=0.35, linewidths=0, rasterized=True)
                counts = []
                for label, style in ERROR_STYLE.items():
                    sel = mask & (labels == label)
                    ax.scatter(xy[sel, 0], xy[sel, 1], marker=style["marker"], c=style["color"], s=16,
                               edgecolors="white", linewidths=0.4, zorder=3)
                    counts.append(int(sel.sum()))
                ax.set_title(f"{group}\n{counts[0]} false alarms, {counts[1]} missed calls", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines[["top", "right", "left", "bottom"]].set_color("0.8")
            axes[i, 0].set_ylabel(name, fontsize=11)
        handles = [plt.Line2D([], [], marker="o", ls="", color=LABEL_COLOURS[0], label="test clip: no upcall", ms=7),
                   plt.Line2D([], [], marker="o", ls="", color=LABEL_COLOURS[1], label="test clip: upcall", ms=7)]
        handles += [plt.Line2D([], [], marker=s["marker"], ls="", color=s["color"], markeredgecolor="white",
                               label=s["label"], ms=8) for s in ERROR_STYLE.values()]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
        fig.suptitle(f"{title}: the section-3 maps (fitted on 28-29 March, 3,000 clips each); all "
                     f"{len(labels):,} test-day clips (31 March) projected in.\nMistakes at each model's "
                     "validation-day 90%-recall threshold drawn on top.", fontsize=11)
        fig.tight_layout(rect=(0, 0.04, 1, 0.94))
        fig.savefig(FIGURES[kind], dpi=130)
        plt.close(fig)
    print("wrote", *FIGURES.values())


if __name__ == "__main__":
    main()
