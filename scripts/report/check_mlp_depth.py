"""Read-only: summary of the fold 0 MLP depth sweep (group perch_mlp_depth), from W&B.

    $env:WANDB_BASE_URL = "https://api.forge.coreweave.com"
    python scripts/report/check_mlp_depth.py
"""
import wandb

KEYS = ["epoch", "train/loss", "val/loss", "val/average_precision"]
api = wandb.Api()
runs = sorted(api.runs("roisegel-personal/whale-upcall", filters={"group": "perch_mlp_depth"}),
              key=lambda r: r.config.get("depth", 0))
print("depth  params    minValLoss@ep  valLoss@10  @20    @40    AP@7   AP@20  AP@40  trainLoss@40")
for r in runs:
    h = {row["epoch"]: row for row in r.scan_history(keys=KEYS)}
    best = min(h.values(), key=lambda row: row["val/loss"])
    g = lambda e, k: h[e][k] if e in h else float("nan")
    print(f"{r.config.get('depth'):>5}  {r.summary.get('trainable_parameters', 0):>8,}  "
          f"{best['val/loss']:.4f}@{best['epoch']:<3}    {g(10, 'val/loss'):.4f}      {g(20, 'val/loss'):.4f} "
          f"{g(40, 'val/loss'):.4f} {g(7, 'val/average_precision'):.4f} {g(20, 'val/average_precision'):.4f} "
          f"{g(40, 'val/average_precision'):.4f} {g(40, 'train/loss'):.4f}")
