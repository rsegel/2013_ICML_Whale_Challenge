"""Read-only: per-epoch train/val loss and val AP of the perch_mlp runs on fold 0 (full data).

    $env:WANDB_BASE_URL = "https://api.forge.coreweave.com"
    python scripts/report/check_mlp_fold0.py
"""
import wandb

api = wandb.Api()
for r in api.runs("roisegel-personal/whale-upcall", filters={"group": "perch_mlp", "config.fold": 0}):
    if r.config.get("train_fraction", 1.0) < 1:
        continue
    print(f"\n{r.name}  best_epoch={r.summary.get('best_epoch')}  "
          f"best_val_AP={r.summary.get('best_val_average_precision')}")
    print("epoch  train/loss  val/loss  val/AP")
    for h in r.scan_history(keys=["epoch", "train/loss", "val/loss", "val/average_precision"]):
        print(f"{h['epoch']:>5}  {h['train/loss']:.4f}      {h['val/loss']:.4f}    {h['val/average_precision']:.4f}")
