"""Label-efficiency figure: average precision against the number of training clips.

Fold 0 only: validation day 30 March (default) or, with ``--split test``, test day
31 March. Every run was trained and early-stopped / tuned on the validation day; the
test scores were saved at training time, so ``--split test`` only scores them. For 10% and 1% of the training clips there
are five runs per model (seeds 0-4); seed s is the same class-stratified subsample
for every model, so the dots line up clip-for-clip. The full-data point is one run
(seed 0). Each run is a dot; the line joins the means.

Reads saved predictions only, of the chosen split. Writes
``reports/figures/label_efficiency_<split>.png`` and ``reports/label_efficiency_<split>.csv``
(one row per run).

Usage::

    python -m whale.figure_label_efficiency            # write the PNG
    python -m whale.figure_label_efficiency --wandb    # also log it to W&B
    python -m whale.figure_label_efficiency --split test   # test day 31 March (asked for 15 Nov)
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from whale import paths
from whale.metrics import ranking_metrics

DAY = {"val": "validation day 30 March", "test": "test day 31 March"}
LABEL = {"val": "validation", "test": "test-day"}


def figure_path(split: str, suffix: str = ""):
    return paths.REPO_ROOT / "reports" / "figures" / f"label_efficiency_{split}{suffix}.png"
PREDICTIONS = paths.CACHE_DIR / "predictions"
FOLDERS = (".", "from_gpu")  # later folder wins: cluster runs replace laptop runs of the same name
FOLD = 0
FRACTIONS = (0.01, 0.1, 1.0)

# Categorical slots 1-3 of the report palette, in fixed order; same entity, same
# colour as the evaluation figures (Perch blue, ResNet orange). Marker shape is a
# second cue so identity never rests on colour alone.
MODELS = {
    "perch_logreg": ("Perch + logistic probe", "#2a78d6", "o"),
    "perch_logreg_whiten": ("Perch, whitened input", "#1baf7a", "^"),
    "resnet18_finetune": ("ResNet-18 fine-tuned", "#eb6834", "s"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def load_runs(split: str) -> pd.DataFrame:
    files = {}
    for folder in FOLDERS:
        files |= {f.name: f for f in sorted((PREDICTIONS / folder).glob(f"*_fold{FOLD}_seed*.parquet"))}
    rows = []
    for file in files.values():
        p = pd.read_parquet(file)
        model, fraction = p["model"].iloc[0], float(p["train_fraction"].iloc[0])
        if model not in MODELS or fraction not in FRACTIONS:
            continue
        part = p[p["split"] == split]
        rows.append({"model": model, "fraction": fraction, "seed": int(p["seed"].iloc[0]),
                     "ap": ranking_metrics(part["label"].to_numpy(), part["score"].to_numpy())["average_precision"]})
    runs = pd.DataFrame(rows)
    full = runs["fraction"] == 1.0
    runs = pd.concat([runs[~full], runs[full & (runs["seed"] == 0)]])  # one full-data run per model
    assert not runs.duplicated(["model", "fraction", "seed"]).any()
    return runs


def plot(runs: pd.DataFrame, n_train: int, sizes: dict[float, int], split: str, suffix: str = "") -> None:
    """``sizes``: actual training clips per fraction (stratified rounding, e.g. 2,270 at 10%)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    offsets = dict(zip(MODELS, (-0.045, 0.0, 0.045)))  # small horizontal nudge, in log10 units
    for model, (label, colour, marker) in MODELS.items():
        part = runs[runs["model"] == model]
        if part.empty:
            continue
        x_of = lambda f: f * n_train * 10 ** offsets[model]
        ax.scatter(x_of(part["fraction"]), part["ap"], s=30, marker=marker, color=colour, alpha=0.45,
                   edgecolor="white", linewidth=0.8, zorder=3)
        means = part.groupby("fraction")["ap"].mean()
        ax.plot(x_of(means.index.to_numpy()), means.to_numpy(), color=colour, linewidth=2, zorder=4)
        ax.scatter(x_of(means.index.to_numpy()), means.to_numpy(), s=64, marker=marker, color=colour,
                   edgecolor="white", linewidth=1.6, zorder=5, label=label)
        # direct label at the 1% end, where the lines are furthest apart
        ax.annotate(f"{label}  {means.iloc[0]:.3f}", (x_of(means.index[0]), means.iloc[0]),
                    xytext=(-16, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=8.5, color=INK)

    ticks = [f * n_train for f in FRACTIONS]
    ax.set_xscale("log")
    ax.set_xticks(ticks, [f"{sizes[f]:,}\n({f:.0%})" for f in FRACTIONS])
    ax.minorticks_off()
    ax.set_xlim(ticks[0] / 12, ticks[-1] * 1.6)
    ax.set_ylim(min(0.62, runs["ap"].min() - 0.02), max(0.95, runs["ap"].max() + 0.02))
    ax.set_xlabel("training clips (share of the full training set)", color=MUTED)
    ax.set_ylabel(f"{LABEL[split]} average precision", color=MUTED)
    ax.set_title(f"Label efficiency, fold 0 ({DAY[split]})", loc="left", fontsize=11,
                 color=INK, pad=20)
    ax.text(0, 1.015, "small marks: one run per subsample (5 seeds; 1 run at 100%)   large marks and lines: mean",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    figure_path(split, suffix).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path(split, suffix), dpi=150)
    plt.close(fig)


def main(log_wandb: bool, split: str = "val", two_model: bool = False) -> None:
    from whale.data.splits import load_labelled
    from whale.train import subsample_training_rows

    manifest = load_labelled(FOLD)
    train_rows = manifest.index[manifest["split"] == "train"].to_numpy()
    sizes = {f: len(subsample_training_rows(manifest, train_rows, f, 0)) for f in FRACTIONS}
    runs = load_runs(split)
    print(runs.pivot_table(index=["fraction", "seed"], columns="model", values="ap").round(3).to_string())
    print("training clips per fraction:", sizes)
    plot(runs, len(train_rows), sizes, split)
    runs.sort_values(["model", "fraction", "seed"]).to_csv(
        paths.REPO_ROOT / "reports" / f"label_efficiency_{split}.csv", index=False)
    print("figure ->", figure_path(split))
    if two_model:  # section-4 version: the two final models only, same runs
        plot(runs[runs["model"] != "perch_logreg"], len(train_rows), sizes, split, "_2model")
        print("figure ->", figure_path(split, "_2model"))
    if log_wandb:
        import wandb

        from whale.train import PROJECT, wandb_mode
        run = wandb.init(project=PROJECT, name=f"label_efficiency_{split}", job_type="eval", mode=wandb_mode())
        run.log({"label_efficiency": wandb.Image(str(figure_path(split))),
                 "label_efficiency_runs": wandb.Table(dataframe=runs)})
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wandb", action="store_true", help="also log the figure and run table to W&B")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--two-model", action="store_true",
                        help="also write a figure without plain Perch (whitened Perch vs ResNet, as in section 4)")
    args = parser.parse_args()
    main(args.wandb, args.split, args.two_model)
