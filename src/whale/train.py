"""Train the spectrogram CNN on one cross-validation fold.

Trains on the fold's training days, picks the epoch with the best validation
average precision, then writes that model's scores for the validation and test
days. The test day is only scored, never used for any decision.

The full training state is saved after every epoch, so a run that is killed
(reboot, power loss) continues from its last finished epoch when the same
command is run again, in the same W&B run.

Usage::

    python -m whale.train --config configs/baseline.yaml --fold 0
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from torch import nn
from torch.utils.data import DataLoader

from whale import paths
from whale.data.dataset import ClipDataset
from whale.data.splits import FOLDS, load_labelled
from whale.metrics import ranking_metrics
from whale.models.resnet import SpectrogramResNet

PROJECT = "whale-upcall"


def wandb_mode() -> str | None:
    """Log online when credentials exist, otherwise offline (sync later with
    ``wandb sync``). An explicit WANDB_MODE always wins."""
    import os

    if os.environ.get("WANDB_MODE"):
        return None
    return "online" if wandb.api.api_key else "offline"


def warmup_cosine(step: int, warmup_steps: int, total_steps: int) -> float:
    """Learning-rate multiplier: linear rise to 1, then half a cosine down to 0."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (manifest rows, labels, logits) for every clip in ``loader``."""
    model.eval()
    rows, labels, logits = [], [], []
    for waveform, label, row in loader:
        logits.append(model(waveform.to(next(model.parameters()).device)).cpu().numpy())
        labels.append(label.numpy())
        rows.append(row.numpy())
    return np.concatenate(rows), np.concatenate(labels), np.concatenate(logits)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def subsample_training_rows(manifest: pd.DataFrame, rows: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Keep a class-stratified ``fraction`` of the training rows."""
    if fraction >= 1:
        return rows
    rng = np.random.default_rng(seed)
    labels = manifest.loc[rows, "label"].to_numpy()
    keep = [rng.choice(rows[labels == c], max(int(round(fraction * (labels == c).sum())), 1), replace=False)
            for c in (0, 1)]
    return np.sort(np.concatenate(keep))


def save_resume_state(file: Path, state: dict) -> None:
    """Write to a temporary file first, so a kill mid-write cannot corrupt the state."""
    temporary = file.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(file)


def run(config: dict, fold: int, limit: int | None = None, resume: bool = True) -> Path:
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(config["threads"])

    run_name = f"{config['name']}_fold{fold}_seed{seed}"
    if config["data"]["train_fraction"] < 1:
        run_name += f"_frac{config['data']['train_fraction']:g}"
    if limit:  # keep smoke tests away from the real run's files
        run_name += f"_limit{limit}"
    checkpoint =paths.CACHE_DIR / "checkpoints" / f"{run_name}.pt"  # best model's weights
    resume_file = checkpoint.with_suffix(".resume.pt")  # full state after the last finished epoch
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = (torch.load(resume_file, weights_only=False, map_location=device)
             if resume and resume_file.exists() else None)
    start_epoch = state["epoch"] if state else 0

    manifest = load_labelled(fold)
    waveforms = np.load(paths.waveform_path("train"))  # 383 MB, held in memory
    rows = {split: manifest.index[manifest["split"] == split].to_numpy() for split in ("train", "val", "test")}
    rows["train"] = subsample_training_rows(manifest, rows["train"], config["data"]["train_fraction"], seed)
    if limit:  # smoke tests
        rows = {split: r[:limit] for split, r in rows.items()}

    batch_size = config["data"]["batch_size"]
    # A resumed run draws new shuffles and shifts rather than replaying the first epochs'.
    train_set = ClipDataset(manifest, waveforms, rows["train"], config["data"]["max_shift"], seed + start_epoch)
    train_loader = DataLoader(train_set, batch_size, shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(seed + start_epoch))
    eval_loaders = {split: DataLoader(ClipDataset(manifest, waveforms, rows[split]), 256)
                    for split in ("val", "test")}

    model = SpectrogramResNet(**config["model"]).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim_cfg = config["optim"]
    optimizer = torch.optim.AdamW(trainable, lr=optim_cfg["lr"], weight_decay=optim_cfg["weight_decay"])
    steps_per_epoch = len(train_loader)
    total_steps = optim_cfg["epochs"] * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: warmup_cosine(s, optim_cfg["warmup_epochs"] * steps_per_epoch, total_steps))
    loss_fn = nn.BCEWithLogitsLoss()

    best_ap, best_epoch, step, elapsed_before = -1.0, -1, 0, 0.0
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best_ap, best_epoch, step = state["best_ap"], state["best_epoch"], state["step"]
        elapsed_before = state["elapsed_seconds"]
        print(f"resuming {run_name} after epoch {start_epoch} (best so far: epoch {best_epoch})", flush=True)

    wandb.init(project=PROJECT, name=run_name, group=config["name"], job_type="train", mode=wandb_mode(),
               id=state["wandb_id"] if state else None, resume="allow" if state else None,
               config={**config, "device": str(device), "fold": fold, "fold_days": FOLDS[fold],
                       "n_train": len(rows["train"]), "n_val": len(rows["val"]),
                       "trainable_parameters": sum(p.numel() for p in trainable),
                       "total_parameters": sum(p.numel() for p in model.parameters())})

    wandb_id = wandb.run.id
    started = time.perf_counter() - elapsed_before

    for epoch in range(start_epoch, optim_cfg["epochs"]):
        model.train()
        running, seen = 0.0, 0
        for waveform, label, _ in train_loader:
            loss = loss_fn(model(waveform.to(device)), label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            running += loss.item() * len(label)
            seen += len(label)
            if step % 25 == 0:
                wandb.log({"train/loss_step": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=step)

        _, val_labels, val_logits = predict(model, eval_loaders["val"])
        val = ranking_metrics(val_labels, sigmoid(val_logits))
        val_loss = loss_fn(torch.from_numpy(val_logits), torch.from_numpy(val_labels)).item()
        wandb.log({"epoch": epoch + 1, "train/loss": running / seen, "val/loss": val_loss,
                   **{f"val/{k}": v for k, v in val.items()},
                   "elapsed_minutes": (time.perf_counter() - started) / 60}, step=step)
        print(f"epoch {epoch + 1:>2}  train loss {running / seen:.4f}  val loss {val_loss:.4f}  "
              f"val AUC {val['roc_auc']:.4f}  val AP {val['average_precision']:.4f}  "
              f"{(time.perf_counter() - started) / 60:.1f} min", flush=True)

        if val["average_precision"] > best_ap:
            best_ap, best_epoch = val["average_precision"], epoch + 1
            torch.save(model.state_dict(), checkpoint)
        elif epoch + 1 - best_epoch >= optim_cfg["patience"]:
            print(f"early stop: no improvement since epoch {best_epoch}")
            break
        save_resume_state(resume_file, {
            "epoch": epoch + 1, "step": step, "best_ap": best_ap, "best_epoch": best_epoch,
            "elapsed_seconds": time.perf_counter() - started, "wandb_id": wandb_id,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()})

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    frames = []
    for split, loader in eval_loaders.items():
        split_rows, labels, logits = predict(model, loader)
        frames.append(pd.DataFrame({
            "clip_index": manifest.loc[split_rows, "clip_index"].to_numpy(), "row": split_rows,
            "split": split, "label": labels.astype(int), "score": sigmoid(logits)}))
    predictions = pd.concat(frames).assign(model=config["name"], fold=fold, seed=seed,
                                           train_fraction=config["data"]["train_fraction"])
    out = paths.CACHE_DIR / "predictions" / f"{run_name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(out, index=False)

    wandb.summary.update({"best_epoch": best_epoch, "best_val_average_precision": best_ap,
                          "train_minutes": (time.perf_counter() - started) / 60})
    wandb.finish()
    resume_file.unlink(missing_ok=True)
    print("predictions ->", out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=0, choices=list(FOLDS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train-fraction", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit", type=int, help="use only the first N clips of each split (smoke test)")
    parser.add_argument("--threads", type=int, help="CPU threads (default: from the config)")
    parser.add_argument("--no-resume", action="store_true", help="ignore saved state and start from epoch 1")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.train_fraction is not None:
        cfg["data"]["train_fraction"] = args.train_fraction
    if args.epochs is not None:
        cfg["optim"]["epochs"] = args.epochs
    if args.threads is not None:
        cfg["threads"] = args.threads
    run(cfg, args.fold, args.limit, resume=not args.no_resume)
