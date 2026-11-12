"""Error analysis on the validation days: what each model gets wrong, and what
the threshold choice costs.

Validation days only. The test days are deliberately not read here.

Prints and saves:

* operating points: false alarms and precision at several recall targets,
  per fold (threshold set on the same day, so these are optimistic);
* error rates split by clip length (short clips are the known weak spot);
* how the two models' errors overlap;
* a gallery of each model's most confident mistakes on the design fold's
  validation day.

Usage::

    python -m whale.analyse_errors
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from whale import paths
from whale.data.preprocess import centre_pad, clean_clip, fade_edges
from whale.data.splits import DESIGN_FOLD
from whale.evaluate import load_predictions
from whale.features.spectrogram import LogSpectrogram
from whale.metrics import operating_point, threshold_for_recall

MODELS = ("perch_logreg", "resnet18_finetune")
FOLDERS = [".", "from_gpu"]
RECALLS = (0.80, 0.90, 0.95, 0.99)
REPORTS = paths.REPO_ROOT / "reports"


def operating_points(val: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, fold), g in val.groupby(["model", "fold"]):
        labels, scores = g["label"].to_numpy(), g["score"].to_numpy()
        for target in RECALLS:
            point = operating_point(labels, scores, threshold_for_recall(labels, scores, target))
            rows.append({"model": model, "fold": fold, "day": g["date"].iloc[0], "recall_target": target,
                         "missed_calls": point["fn"], "false_alarms": point["fp"],
                         "false_alarms_per_1000_clips": round(point["false_positives_per_1000_clips"], 1),
                         "precision": round(point["precision"], 3)})
    return pd.DataFrame(rows)


def flag_errors(val: pd.DataFrame, recall: float = 0.90) -> pd.DataFrame:
    """Add ``wrong`` at each (model, fold)'s own threshold for ``recall``."""
    parts = []
    for _, g in val.groupby(["model", "fold"]):
        threshold = threshold_for_recall(g["label"].to_numpy(), g["score"].to_numpy(), recall)
        parts.append(g.assign(flagged=g["score"] >= threshold))
    out = pd.concat(parts)
    out["wrong"] = out["flagged"] != (out["label"] == 1)
    out["short"] = out["n_samples"] < paths.CLIP_SAMPLES
    return out


def by_length(errors: pd.DataFrame) -> pd.DataFrame:
    """Miss rate among upcalls and false-alarm rate among non-calls, short vs full clips."""
    table = errors.groupby(["model", "fold", "short", "label"])["wrong"].agg(["mean", "size"]).reset_index()
    table["kind"] = np.where(table["label"] == 1, "missed_share_of_calls", "false_alarm_share_of_non_calls")
    return table.rename(columns={"mean": "rate", "size": "clips"}).drop(columns="label")


def overlap(errors: pd.DataFrame) -> pd.DataFrame:
    wide = errors.pivot(index=["fold", "row", "label"], columns="model", values="wrong").reset_index()
    a, b = MODELS
    wide["who"] = np.select([wide[a] & wide[b], wide[a], wide[b]], ["both", f"only {a}", f"only {b}"], "neither")
    wide["kind"] = np.where(wide["label"] == 1, "missed call", "false alarm")
    return wide[wide["who"] != "neither"].groupby(["fold", "kind", "who"]).size().unstack(fill_value=0)


def gallery(errors: pd.DataFrame, file, per_kind: int = 6) -> None:
    """Most confident mistakes of each model on the design fold's validation day."""
    part = errors[errors["fold"] == DESIGN_FOLD]
    waveforms = np.load(paths.waveform_path("train"), mmap_mode="r")
    to_spectrogram = LogSpectrogram()
    fig, axes = plt.subplots(4, per_kind, figsize=(2.6 * per_kind, 9.5), sharex=True, sharey=True)
    row_specs = [(m, kind) for m in MODELS for kind in ("false alarm", "missed call")]
    for axis_row, (model, kind) in zip(axes, row_specs):
        g = part[part["model"] == model]
        picks = (g[g["label"] == 0].nlargest(per_kind, "score") if kind == "false alarm"
                 else g[g["label"] == 1].nsmallest(per_kind, "score"))
        other = part[part["model"] != model].set_index("row")["score"]
        for axis, (_, clip) in zip(axis_row, picks.iterrows()):
            samples = centre_pad(fade_edges(clean_clip(waveforms[clip["row"]], clip["n_samples"])), paths.CLIP_SAMPLES)
            with torch.no_grad():
                image = to_spectrogram(torch.from_numpy(samples)[None])[0, 0].numpy()
            axis.imshow(image, origin="lower", aspect="auto", cmap="magma", extent=(0, 2, 40, 500), vmin=-1, vmax=3)
            axis.set_title(f"score {clip['score']:.2f} (other model {other[clip['row']]:.2f})\n"
                           f"{clip['n_samples'] / paths.SAMPLE_RATE:.1f} s clip", fontsize=8)
        axis_row[0].set_ylabel(f"{model}\n{kind}\nHz", fontsize=9)
    for axis in axes[-1]:
        axis.set_xlabel("seconds")
    fig.suptitle("Most confident mistakes on the design fold's validation day (30 March), as the CNN sees them",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(file, dpi=130)
    plt.close(fig)


def main() -> None:
    pred = load_predictions(list(MODELS), FOLDERS)
    val = pred[pred["split"] == "val"]
    assert set(val["split"]) == {"val"}

    points = operating_points(val)
    print("=== operating points per validation day (threshold set on the same day) ===")
    print(points.to_string(index=False))
    errors = flag_errors(val)
    lengths = by_length(errors)
    print("\n=== error rates at 90% recall, short vs full-length clips ===")
    print(lengths.round(3).to_string(index=False))
    both = overlap(errors)
    print("\n=== who is wrong, at each model's 90%-recall threshold ===")
    print(both.to_string())

    (REPORTS / "error_analysis_val.json").write_text(json.dumps({
        "operating_points": points.to_dict("records"), "by_clip_length": lengths.to_dict("records"),
        "overlap": both.reset_index().to_dict("records")}, indent=2))
    figure = REPORTS / "figures" / "error_gallery_val.png"
    gallery(errors, figure)
    print("\nwrote", REPORTS / "error_analysis_val.json", "and", figure)


if __name__ == "__main__":
    main()
