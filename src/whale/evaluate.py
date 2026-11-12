"""Compare the models on held-out days from their saved predictions.

Every fold has its own model, tested on a day it never saw. Metrics are
computed per fold and then averaged over folds, so each day counts equally and
scores from differently trained models are never ranked against each other.

* Decision thresholds come from the fold's validation day and are applied
  unchanged to the test day, as they would be in deployment.
* Confidence intervals resample whole 15-minute chunks within each day,
  because neighbouring clips are correlated. Both models are scored on the same
  resamples, so the interval of their difference is paired.

``--split val`` scores the validation days instead. It exists so that this
script could be written and debugged without reading any test-day result; its
numbers are optimistic (thresholds and model selection used the same day).

Usage::

    python -m whale.evaluate                  # validation days (default)
    python -m whale.evaluate --split test     # held-out test days: run once, at the very end
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from whale import paths
from whale.data.splits import FOLDS
from whale.metrics import operating_point, ranking_metrics, recall_at_fpr, threshold_for_recall

PREDICTIONS = paths.CACHE_DIR / "predictions"
REPORTS = paths.REPO_ROOT / "reports"
RECALL_TARGETS = (0.90, 0.95)
COLOURS = ("#2a78d6", "#d9622b", "#3a9d6e", "#8a5fbf")


def load_predictions(models: list[str] | None = None, folders: list[str] | None = None) -> pd.DataFrame:
    """All full-data runs, one row per (model, fold, clip), with manifest columns.

    ``folders`` are read in order (relative to the predictions directory); when
    two hold the same run, the later folder wins.
    """
    files = {}
    for folder in folders or ["."]:
        files |= {f.name: f for f in sorted((PREDICTIONS / folder).glob("*.parquet"))}
    frames = [pd.read_parquet(f) for f in files.values()]
    pred = pd.concat(frames, ignore_index=True)
    pred = pred[pred["train_fraction"] == 1.0]
    if models:
        pred = pred[pred["model"].isin(models)]
    manifest = pd.read_parquet(paths.manifest_path("train"))[["date", "chunk_id", "n_samples"]]
    pred = pred.join(manifest, on="row")
    duplicated = pred.duplicated(["model", "fold", "seed", "split", "row"])
    assert not duplicated.any(), "the same clip is predicted twice by one run"
    return pred


def common_folds(pred: pd.DataFrame) -> list[int]:
    """Folds for which every model has predictions."""
    per_model = pred.groupby("model")["fold"].agg(set)
    return sorted(set.intersection(*per_model))


def fold_metrics(val: pd.DataFrame, part: pd.DataFrame) -> dict[str, float]:
    """Metrics of one model on one fold. ``val`` supplies the thresholds."""
    labels, scores = part["label"].to_numpy(), part["score"].to_numpy()
    out = {"clips": len(part), "positives": int(labels.sum()), **ranking_metrics(labels, scores),
           "recall_at_1pct_fpr": recall_at_fpr(labels, scores, 0.01)}
    for target in RECALL_TARGETS:
        threshold = threshold_for_recall(val["label"].to_numpy(), val["score"].to_numpy(), target)
        point = operating_point(labels, scores, threshold)
        tag = f"at_{round(100 * target)}"
        out |= {f"recall_{tag}": point["recall"], f"precision_{tag}": point["precision"],
                f"fp_per_1000_{tag}": point["false_positives_per_1000_clips"], f"threshold_{tag}": threshold}
    full = (part["n_samples"] == paths.CLIP_SAMPLES).to_numpy()
    out["roc_auc_full_length"] = ranking_metrics(labels[full], scores[full])["roc_auc"]
    out["average_precision_full_length"] = ranking_metrics(labels[full], scores[full])["average_precision"]
    return out


def per_fold_table(pred: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for (model, fold), group in pred.groupby(["model", "fold"]):
        part = group[group["split"] == split]
        rows.append({"model": model, "fold": fold, "day": FOLDS[fold][split][0],
                     **fold_metrics(group[group["split"] == "val"], part)})
    return pd.DataFrame(rows)


BOOT_METRICS = ("roc_auc", "average_precision", "brier", "recall_at_90", "fp_per_1000_at_90")


def bootstrap(pred: pd.DataFrame, split: str, folds: list[int], n_boot: int, seed: int = 0) -> dict:
    """Chunk bootstrap of fold-averaged metrics, paired across models.

    Returns ``{model: {metric: [low, high]}}`` plus ``{"a - b": {...}}`` for
    every pair of models (95% intervals).
    """
    rng = np.random.default_rng(seed)
    models = sorted(pred["model"].unique())
    data = {}  # fold -> aligned arrays
    for fold in folds:
        in_fold = pred[pred["fold"] == fold]
        wide = in_fold[in_fold["split"] == split].pivot(index="row", columns="model", values="score")
        assert not wide.isna().any().any(), f"fold {fold}: models scored different clips"
        meta = in_fold[in_fold["split"] == split].drop_duplicates("row").set_index("row").loc[wide.index]
        thresholds = {}
        for m in models:
            val = in_fold[(in_fold["split"] == "val") & (in_fold["model"] == m)]
            thresholds[m] = threshold_for_recall(val["label"].to_numpy(), val["score"].to_numpy(), 0.90)
        chunks = meta["chunk_id"].to_numpy()
        data[fold] = {"labels": meta["label"].to_numpy(), "scores": {m: wide[m].to_numpy() for m in models},
                      "groups": [np.flatnonzero(chunks == c) for c in np.unique(chunks)], "thresholds": thresholds}

    draws = {m: {k: [] for k in BOOT_METRICS} for m in models}
    for _ in range(n_boot):
        replicate = {m: {k: [] for k in BOOT_METRICS} for m in models}
        for d in data.values():
            picked = rng.integers(len(d["groups"]), size=len(d["groups"]))
            idx = np.concatenate([d["groups"][i] for i in picked])
            labels = d["labels"][idx]
            for m in models:
                scores = d["scores"][m][idx]
                point = operating_point(labels, scores, d["thresholds"][m])
                values = {**ranking_metrics(labels, scores), "recall_at_90": point["recall"],
                          "fp_per_1000_at_90": point["false_positives_per_1000_clips"]}
                for k in BOOT_METRICS:
                    replicate[m][k].append(values[k])
        for m in models:
            for k in BOOT_METRICS:
                draws[m][k].append(np.mean(replicate[m][k]))

    def interval(x):
        return [float(v) for v in np.percentile(x, [2.5, 97.5])]

    out = {m: {k: interval(draws[m][k]) for k in BOOT_METRICS} for m in models}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            out[f"{a} - {b}"] = {k: interval(np.array(draws[a][k]) - np.array(draws[b][k])) for k in BOOT_METRICS}
    return out


def error_overlap(pred: pd.DataFrame, split: str, folds: list[int]) -> pd.DataFrame:
    """At each model's 90%-recall threshold: which clips do the models get wrong together?"""
    models = sorted(pred["model"].unique())
    wrong = []
    for fold in folds:
        in_fold = pred[pred["fold"] == fold]
        for m in models:
            group = in_fold[in_fold["model"] == m]
            val, part = group[group["split"] == "val"], group[group["split"] == split]
            threshold = threshold_for_recall(val["label"].to_numpy(), val["score"].to_numpy(), 0.90)
            flagged = part["score"] >= threshold
            wrong.append(pd.DataFrame({"row": part["row"], "label": part["label"], "model": m,
                                       "wrong": flagged != (part["label"] == 1)}))
    wide = pd.concat(wrong).pivot_table(index=["row", "label"], columns="model", values="wrong").astype(bool)
    wide = wide.reset_index()
    wide["kind"] = np.where(wide["label"] == 1, "missed call", "false alarm")
    return wide.groupby("kind")[models].apply(
        lambda g: pd.Series({"all models wrong": int(g.all(axis=1).sum()),
                             **{f"only {m} wrong": int((g[m] & ~g.drop(columns=m).any(axis=1)).sum())
                                for m in models}}))


def worst_errors(pred: pd.DataFrame, split: str, n: int = 25) -> pd.DataFrame:
    """Most confident mistakes of each model, for listening and plotting."""
    part = pred[pred["split"] == split]
    frames = []
    for model, group in part.groupby("model"):
        frames.append(group[group["label"] == 0].nlargest(n, "score").assign(kind="false alarm"))
        frames.append(group[group["label"] == 1].nsmallest(n, "score").assign(kind="missed call"))
    return pd.concat(frames)[["model", "kind", "fold", "date", "clip_index", "row", "n_samples", "label", "score"]]


def plot(pred: pd.DataFrame, split: str, folds: list[int], file) -> None:
    """Precision-recall per fold (left) and calibration pooled over folds (right)."""
    models = sorted(pred["model"].unique())
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))
    part = pred[(pred["split"] == split) & pred["fold"].isin(folds)]
    for colour, model in zip(COLOURS, models):
        for i, (fold, group) in enumerate(part[part["model"] == model].groupby("fold")):
            precision, recall, _ = precision_recall_curve(group["label"], group["score"])
            left.plot(recall, precision, color=colour, lw=1.5, alpha=0.75, label=model if i == 0 else None)
        group = part[part["model"] == model]
        bins = pd.cut(group["score"], np.linspace(0, 1, 11), include_lowest=True)
        calibration = group.groupby(bins, observed=True).agg(predicted=("score", "mean"), observed=("label", "mean"))
        right.plot(calibration["predicted"], calibration["observed"], "o-", color=colour, lw=2, ms=6, label=model)
    left.set(xlabel="Recall (share of upcalls caught)", ylabel="Precision (share of alarms that are upcalls)",
             title=f"Precision-recall, one line per {split} day", xlim=(0.5, 1), ylim=(0, 1.02))
    right.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="perfectly calibrated")
    right.set(xlabel="Predicted probability (mean within each 0.1-wide bin)", ylabel="Observed share of upcalls",
              title="Calibration")
    for axis in (left, right):
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(file, dpi=150)
    plt.close(fig)


FOLD_PANELS = (("roc_auc", "ROC-AUC"), ("average_precision", "Average precision"),
               ("fp_per_1000_at_90", "False alarms per 1,000 clips\nat the 90%-recall threshold"),
               ("recall_at_90", "Recall reached at the\n90%-recall threshold"))


def plot_folds(table: pd.DataFrame, intervals: dict, split: str, file) -> None:
    """Every fold's value as a point (same day joined across models), with the
    mean over folds and its 95% interval beside them."""
    models = sorted(table["model"].unique())
    fig, axes = plt.subplots(1, len(FOLD_PANELS), figsize=(3.4 * len(FOLD_PANELS), 4.4))
    for axis, (metric, title) in zip(axes, FOLD_PANELS):
        wide = table.pivot(index="day", columns="model", values=metric)[models]
        for day, values in wide.iterrows():
            axis.plot(range(len(models)), values, color="0.75", lw=1, zorder=1)
            axis.annotate(f"{day[6:]} Mar", (len(models) - 1, values.iloc[-1]), xytext=(6, 0),
                          textcoords="offset points", va="center", fontsize=8, color="0.35")
        for x, (colour, model) in enumerate(zip(COLOURS, models)):
            axis.scatter([x] * len(wide), wide[model], s=42, color=colour, edgecolor="white", lw=1, zorder=2)
            low, high = intervals[model][metric]
            mean = wide[model].mean()
            axis.errorbar(x - 0.22, mean, yerr=[[mean - low], [high - mean]], fmt="D", color=colour,
                          ms=6, capsize=3, lw=1.5, zorder=3)
        axis.set_xticks(range(len(models)), [m.replace("_", "\n") for m in models])
        axis.set_xlim(-0.5, len(models) - 0.3)
        axis.set_title(title, fontsize=10)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Circles: one {split} day each (grey line joins the same day). "
                 "Diamond: mean over days with 95% chunk-bootstrap interval.", fontsize=9, y=0.995)
    fig.tight_layout()
    fig.savefig(file, dpi=150)
    plt.close(fig)


def main(split: str, n_boot: int, models: list[str] | None, log_wandb: bool,
         folders: list[str] | None = None) -> None:
    pred = load_predictions(models, folders)
    folds = common_folds(pred)
    missing = sorted(set(FOLDS) - set(folds))
    if missing:
        print(f"WARNING: folds {missing} are missing for at least one model; comparing on folds {folds} only.")
    pred = pred[pred["fold"].isin(folds)]

    table = per_fold_table(pred, split)
    numeric = [c for c in table.columns if c not in ("model", "fold", "day", "clips", "positives")]
    mean = table.groupby("model")[numeric].mean()
    spread = table.groupby("model")[["roc_auc", "average_precision"]].agg(["min", "max"])
    intervals = bootstrap(pred, split, folds, n_boot)
    overlap = error_overlap(pred, split, folds)

    shown = ["model", "day", "clips", "positives", "roc_auc", "average_precision", "brier",
             "recall_at_90", "fp_per_1000_at_90", "recall_at_95", "fp_per_1000_at_95", "recall_at_1pct_fpr"]
    print(f"\n=== per {split} day ===\n" + table[shown].round(4).to_string(index=False))
    print("\n=== mean over folds, with 95% chunk-bootstrap interval ===")
    for model in mean.index:
        print(model)
        for k in BOOT_METRICS:
            low, high = intervals[model][k]
            raw = ", ".join(f"{v:.4f}" for v in table.loc[table["model"] == model, k])
            print(f"  {k:<20} {mean.loc[model, k]:.4f}  [{low:.4f}, {high:.4f}]   per fold: {raw}")
        print(f"  full-length clips only: roc_auc {mean.loc[model, 'roc_auc_full_length']:.4f}  "
              f"average_precision {mean.loc[model, 'average_precision_full_length']:.4f}")
    print("\n=== paired differences (95% interval; excludes 0 = distinguishable) ===")
    for name, values in intervals.items():
        if " - " in name:
            print(name)
            for k in BOOT_METRICS:
                print(f"  {k:<20} [{values[k][0]:+.4f}, {values[k][1]:+.4f}]")
    print("\n=== errors at each model's 90%-recall threshold ===\n" + overlap.to_string())

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "figures").mkdir(exist_ok=True)
    result = {"split": split, "folds": folds, "n_boot": n_boot, "per_fold": table.to_dict("records"),
              "mean_over_folds": mean.to_dict("index"), "intervals_95": intervals,
              "spread_over_days": {m: {f"{a}_{b}": float(spread.loc[m, (a, b)]) for a, b in spread.columns}
                                   for m in spread.index},
              "error_overlap": overlap.to_dict("index")}
    (REPORTS / f"evaluation_{split}.json").write_text(json.dumps(result, indent=2))
    errors = worst_errors(pred, split)
    errors.to_csv(REPORTS / f"worst_errors_{split}.csv", index=False)
    figure = REPORTS / "figures" / f"evaluation_{split}.png"
    plot(pred, split, folds, figure)
    folds_figure = REPORTS / "figures" / f"evaluation_{split}_folds.png"
    plot_folds(table, intervals, split, folds_figure)
    print(f"\nwrote {REPORTS / f'evaluation_{split}.json'}, worst_errors_{split}.csv, {figure} and {folds_figure}")

    if log_wandb:
        import wandb

        from whale.train import PROJECT
        run = wandb.init(project=PROJECT, name=f"evaluation_{split}", job_type="eval",
                         config={"split": split, "folds": folds, "n_boot": n_boot})
        run.log({"per_fold": wandb.Table(dataframe=table), "worst_errors": wandb.Table(dataframe=errors),
                 "mean_over_folds": wandb.Table(dataframe=mean.reset_index()),
                 "curves": wandb.Image(str(figure)), "per_fold_points": wandb.Image(str(folds_figure))})
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="test is opt-in: read the test days only once, when everything else is final")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--models", nargs="+", help="model names to include (default: all full-data runs)")
    parser.add_argument("--wandb", action="store_true", help="also log the tables and figure to W&B")
    parser.add_argument("--folders", nargs="+", help="sub-folders of the predictions directory, later ones win "
                        "(default: the directory itself)")
    args = parser.parse_args()
    main(args.split, args.n_boot, args.models, args.wandb, args.folders)
