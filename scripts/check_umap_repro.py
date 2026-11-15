"""Why do the section-3 and section-4 UMAPs differ? Training days (28-29 Mar) only; no test data.

1. Re-builds the section-3 inputs (explore_embeddings.sample_clips + load_representations) and compares the
   deterministic summary numbers with reports/embedding_structure.json (12 Nov). Equal numbers = same data.
2. Fits each UMAP twice with the same seed and compares the coordinates (is the fit repeatable at all?).
3. Saves the by-label map of each refit next to nothing else: reports/figures/check_umap_repro.png.

    python scripts/check_umap_repro.py
"""
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from whale import paths
from whale.explore_embeddings import (LABEL_COLOURS, N_COMPONENTS, RESULTS, SEED, linear_separability,
                                      load_representations, sample_clips)

saved = json.loads(RESULTS.read_text())
sample = sample_clips()
reps = load_representations(sample)
order = np.random.default_rng(SEED).permutation(len(sample))
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for row, (name, values) in enumerate(reps.items()):
    std = StandardScaler().fit_transform(values)
    pca = PCA(N_COMPONENTS, random_state=SEED).fit(std)
    pc = pca.transform(std)
    now = {**linear_separability(pc, sample), "variance_in_first_2_components":
           float(pca.explained_variance_ratio_[:2].sum())}
    print(name)
    for k, v in now.items():
        print(f"  {k}: now {v:.6f}  12 Nov {saved[name][k]:.6f}")
    runs = []
    for attempt in range(2):
        t = time.time()
        runs.append(umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=SEED).fit_transform(pc))
        print(f"  UMAP fit {attempt}: {time.time() - t:.0f} s")
    print("  max |difference| between the two fits:", float(np.abs(runs[0] - runs[1]).max()))
    up = sample["label"].to_numpy() == 1
    print("  upcall centre minus non-upcall centre (x, y):", (runs[0][up].mean(0) - runs[0][~up].mean(0)).round(2))
    for col, xy in enumerate(runs):
        ax = axes[row, col]
        ax.scatter(xy[order, 0], xy[order, 1], c=sample["label"].map(LABEL_COLOURS).to_numpy()[order], s=3,
                   alpha=0.55, linewidths=0, rasterized=True)
        ax.set_title(f"{name.split(' (')[0]}, fit {col}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
fig.tight_layout()
out = paths.REPO_ROOT / "reports" / "figures" / "check_umap_repro.png"
fig.savefig(out, dpi=90)
print("wrote", out)
