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
from whale.metrics import ranking_metrics
from whale.probe import INPUT
from whale.train import PROJECT, sigmoid, subsample_training_rows, wandb_mode

NAME = "perch_mlp"
CONFIG = {"hidden": 256, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-2, "batch_size": 256,
          "epochs": 40, "patience": 5}


def run(fold: int, seed: int = 0, train_fraction: float = 1.0):
    torch.manual_seed(seed)
    manifest = load_labelled(fold)
    embeddings = np.load(embedding_path("train", INPUT), mmap_mode="r")
    rows = {split: manifest.index[manifest["split"] == split].to_numpy() for split in ("train", "val", "test")}
    rows["train"] = subsample_training_rows(manifest, rows["train"], train_fraction, seed)
    features = {split: np.asarray(embeddings[r]) for split, r in rows.items()}
    mean, std = features["train"].mean(0), features["train"].std(0) + 1e-6  # scaling from training days only
    x = {split: torch.from_numpy((f - mean) / std) for split, f in features.items()}
    y = {split: torch.from_numpy(manifest.loc[r, "label"].to_numpy(dtype=np.float32)) for split, r in rows.items()}

    model = nn.Sequential(nn.Linear(x["train"].shape[1], CONFIG["hidden"]), nn.ReLU(), nn.Dropout(CONFIG["dropout"]),
                          nn.Linear(CONFIG["hidden"], 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    run_name = f"{NAME}_fold{fold}_seed{seed}" + (f"_frac{train_fraction:g}" if train_fraction < 1 else "")
    wandb.init(project=PROJECT, name=run_name, group=NAME, job_type="train", mode=wandb_mode(),
               config={**CONFIG, "name": NAME, "fold": fold, "fold_days": FOLDS[fold], "seed": seed,
                       "input": INPUT.name, "train_fraction": train_fraction, "n_train": len(rows["train"]),
                       "trainable_parameters": sum(p.numel() for p in model.parameters())})

    def logits(split: str) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            return model(x[split]).squeeze(1)

    best_ap, best_epoch, best_state, started = -1.0, -1, None, time.perf_counter()
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(CONFIG["epochs"]):
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
        elif epoch + 1 - best_epoch >= CONFIG["patience"]:
            break
    print(f"fold {fold} frac {train_fraction:g} seed {seed}: best epoch {best_epoch} of {epoch + 1}, "
          f"val AP {best_ap:.4f}, {time.perf_counter() - started:.0f} s", flush=True)

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
    args = parser.parse_args()
    run(args.fold, args.seed, args.train_fraction)
