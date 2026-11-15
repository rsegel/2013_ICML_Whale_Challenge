"""Mean spectrogram of the test-day mistakes, fold 0 (test day 31 March).

Errors are taken at each model's 90%-recall threshold from the fold's validation
day (as in ``evaluate.error_overlap``), for whitened Perch and ResNet-18:

* rows: false alarms (label 0, flagged) and missed calls (label 1, not flagged);
* columns: wrong only for whitened Perch, wrong only for ResNet, wrong for both,
  and, for reference, the clips both models got right (true negatives / true positives).

Each panel is the mean over clips of a per-clip whitened log spectrogram: the clip's
offset and level are removed (``clean_clip``), then each frequency's median dB over
time is subtracted. Whitening removes the day's noise floor so the mean shows
structure rather than loudness. Only full-length (2 s) clips are averaged; short
clips are counted in the title.

    python -m whale.figure_test_error_spectrograms
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from whale import paths
from whale.data.audit import spectrogram
from whale.data.preprocess import clean_clip
from whale.evaluate import load_predictions
from whale.metrics import threshold_for_recall

FOLD = 0
MODELS = {"perch_logreg_whiten": "whitened Perch", "resnet18_finetune": "ResNet-18"}
OUT = paths.REPO_ROOT / "reports" / "figures" / "test_error_spectrograms_fold0.png"


def wrong_table() -> pd.DataFrame:
    pred = load_predictions(list(MODELS), [".", "from_gpu"])
    pred = pred[pred["fold"] == FOLD]
    cols = {}
    for m in MODELS:
        group = pred[pred["model"] == m]
        val, test = group[group["split"] == "val"], group[group["split"] == "test"]
        threshold = threshold_for_recall(val["label"].to_numpy(), val["score"].to_numpy(), 0.90)
        test = test.set_index("row")
        cols[m] = (test["score"] >= threshold) != (test["label"] == 1)
        cols["label"], cols["n_samples"] = test["label"], test["n_samples"]
    return pd.DataFrame(cols)


def whitened_db(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs, times, db = spectrogram(clean_clip(samples, paths.CLIP_SAMPLES))
    return freqs, times, db - np.median(db, axis=1, keepdims=True)


def main() -> None:
    table = wrong_table()
    waveforms = np.load(paths.waveform_path("train"), mmap_mode="r")
    p, r = table["perch_logreg_whiten"], table["resnet18_finetune"]
    groups = {"only whitened Perch wrong": p & ~r, "only ResNet wrong": ~p & r,
              "both wrong": p & r, "both right (reference)": ~p & ~r}
    kinds = {0: "False alarms", 1: "Missed calls"}
    reference_names = {0: "true negatives", 1: "true positives"}

    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for i, (label, kind) in enumerate(kinds.items()):
        means, titles = [], []
        for name, mask in groups.items():
            sel = table[mask & (table["label"] == label)]
            full = sel.index[sel["n_samples"] == paths.CLIP_SAMPLES].to_numpy()
            n_short = len(sel) - len(full)
            sampled = name.startswith("both right") and len(full) > 2000
            if sampled:
                full = rng.choice(full, 2000, replace=False)  # reference only; a sample is enough
            stack = [whitened_db(waveforms[row])[2] for row in full]
            freqs, times, _ = whitened_db(waveforms[full[0]])
            means.append(np.mean(stack, axis=0))
            shown = name.replace("(reference)", f"({reference_names[label]})")
            titles.append(f"{kind}: {shown}\nn = {len(sel)}"
                          + (f" ({n_short} short clips left out)" if n_short else "")
                          + (f", mean of {len(full)} sampled" if sampled else ""))
        vmin = min(m.min() for m in means)
        vmax = max(m.max() for m in means)
        for ax, mean, title in zip(axes[i], means, titles):
            mesh = ax.pcolormesh(times, freqs, mean, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9)
        fig.colorbar(mesh, ax=axes[i].tolist(), label="dB above the clip's\nper-frequency median", pad=0.01)
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("frequency (Hz)")
    fig.suptitle("Fold 0 test day (31 March): mean whitened spectrogram of the mistakes at each model's "
                 "validation-day 90%-recall threshold (shared colour scale per row)", fontsize=11)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)

    counts = {k: {kinds[l]: int((m & (table["label"] == l)).sum()) for l in kinds} for k, m in groups.items()}
    print(pd.DataFrame(counts).to_string())
    print("saved", OUT)


if __name__ == "__main__":
    main()
