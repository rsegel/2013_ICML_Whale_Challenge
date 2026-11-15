"""Train a small neural-network probe on frozen Perch embeddings for one fold.

Same protocol as ``whale.train``: fit on the fold's training days, keep the
epoch with the best validation average precision, write that model's scores for
the validation and test days. Unlike the logistic probe it is trained by
gradient descent, so it has loss curves to set beside the CNN's in W&B.

The architecture and optimiser settings are fixed defaults, not tuned.

Usage::

    python -m whale.probe_mlp --fold 0
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
import wandb
from torch import nn

from whale import paths
from whale.data.splits import FOLDS, load_labelled
from whale.embed import embedding_path
from whale.features.perch_input import PerchInputConfig
from whale.metrics import ranking_metrics
from whale.probe import INPUT
from whale.train import PROJECT, sigmoid, subsample_training_rows, wandb_mode

NAME = "perch_mlp"
CONFIG = {"hidden": 256, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-2, "batch_size": 256,
          "epochs": 40, "patience": 5}


def run(fold: int, seed: int = 0, train_fraction: float = 1.0, depth: int | None = None, epochs: int | None = None,
        denoise: str = "none"):
    """``depth``/``epochs`` set: depth sweep. Hidden layers of the same width, no early stopping, group
    ``perch_mlp_depth``. ``denoise`` set: same MLP and early stopping on the denoised-input embeddings,
    group ``perch_mlp_<denoise>``. Both are analysis runs: no test-day rows are loaded or scored."""
    sweep = depth is not None
    analysis = sweep or denoise != "none"
    config = {**CONFIG, "depth": depth or 1, "denoise": denoise}
    if sweep:
        config.update(epochs=epochs or CONFIG["epochs"], patience=10**9)
    input_config = INPUT if denoise == "none" else PerchInputConfig(INPUT.speed, INPUT.pad, INPUT.peak, denoise)
    torch.manual_seed(seed)
    manifest = load_labelled(fold)
    embeddings = np.load(embedding_path("train", input_config), mmap_mode="r")
    assert len(embeddings) == len(manifest), "embedding file does not match the manifest"
    splits = ("train", "val") if analysis else ("train", "val", "test")
    rows = {split: manifest.index[manifest["split"] == split].to_numpy() for split in splits}
    rows["train"] = subsample_training_rows(manifest, rows["train"], train_fraction, seed)
    features = {split: np.asarray(embeddings[r]) for split, r in rows.items()}
    mean, std = features["train"].mean(0), features["train"].std(0) + 1e-6  # scaling from training days only
    x = {split: torch.from_numpy((f - mean) / std) for split, f in features.items()}
    y = {split: torch.from_numpy(manifest.loc[r, "label"].to_numpy(dtype=np.float32)) for split, r in rows.items()}

    layers, width = [], x["train"].shape[1]
    for _ in range(config["depth"]):
        layers += [nn.Linear(width, config["hidden"]), nn.ReLU(), nn.Dropout(config["dropout"])]
        width = config["hidden"]
    model = nn.Sequential(*layers, nn.Linear(width, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    name = NAME + (f"_{denoise}" if denoise != "none" else "") + ("_depth" if sweep else "")
    run_name = name + (str(depth) if sweep else "") + f"_fold{fold}_seed{seed}" \
        + (f"_frac{train_fraction:g}" if train_fraction < 1 else "")
    wandb.init(project=PROJECT, name=run_name, group=name, job_type="train", mode=wandb_mode(),
               config={**config, "name": name, "fold": fold, "fold_days": FOLDS[fold], "seed": seed,
                       "input": input_config.name, "train_fraction": train_fraction, "n_train": len(rows["train"]),
                       "trainable_parameters": sum(p.numel() for p in model.parameters())})

    def logits(split: str) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            return model(x[split]).squeeze(1)

    best_ap, best_epoch, best_state, started = -1.0, -1, None, time.perf_counter()
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(config["epochs"]):
        model.train()
        order = torch.randperm(len(x["train"]), generator=generator)
        running = 0.0
        for start in range(0, len(order), CONFIG["batch_size"]):
            batch = order[start : start + CONFIG["batch_size"]]
            loss = loss_fn(model(x["train"][batch]).squeeze(1), y["train"][batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(batch)
        val_logits = logits("val")
        val = ranking_metrics(y["val"].numpy(), sigmoid(val_logits.numpy()))
        wandb.log({"epoch": epoch + 1, "train/loss": running / len(order),
                   "val/loss": loss_fn(val_logits, y["val"]).item(), **{f"val/{k}": v for k, v in val.items()}})
        if val["average_precision"] > best_ap:
            best_ap, best_epoch = val["average_precision"], epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch + 1 - best_epoch >= config["patience"]:
            break
    print(f"{run_name}: best epoch {best_epoch} of {epoch + 1}, "
          f"val AP {best_ap:.4f}, {time.perf_counter() - started:.0f} s", flush=True)

    if analysis:  # curves only, no predictions file, no test rows
        wandb.summary.update({"best_epoch": best_epoch, "best_val_average_precision": best_ap,
                              "trainable_parameters": sum(p.numel() for p in model.parameters())})
        wandb.finish()
        return None

    model.load_state_dict(best_state)
    frames = [pd.DataFrame({
        "clip_index": manifest.loc[rows[split], "clip_index"].to_numpy(), "row": rows[split], "split": split,
        "label": y[split].numpy().astype(int), "score": sigmoid(logits(split).numpy())}) for split in ("val", "test")]
    out = paths.CACHE_DIR / "predictions" / f"{run_name}.parquet"
    pd.concat(frames).assign(model=NAME, fold=fold, seed=seed, train_fraction=train_fraction).to_parquet(out, index=False)
    wandb.summary.update({"best_epoch": best_epoch, "best_val_average_precision": best_ap,
                          "train_minutes": (time.perf_counter() - started) / 60})
    wandb.finish()
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fold", type=int, default=0, choices=list(FOLDS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--depth", type=int, help="depth sweep: number of hidden layers (no early stopping, "
                                                  "validation only, group perch_mlp_depth)")
    parser.add_argument("--epochs", type=int, help="with --depth: epochs to train (default 40)")
    parser.add_argument("--denoise", choices=["none", "whiten", "subtract"], default="none",
                        help="use embeddings of the denoised Perch input (validation only, group perch_mlp_<denoise>)")
    args = parser.parse_args()
    run(args.fold, args.seed, args.train_fraction, args.depth, args.epochs, args.denoise)
