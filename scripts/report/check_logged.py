"""Read-only: list what W&B actually holds for the two things the §2 panels need.

    $env:WANDB_BASE_URL = "https://api.forge.coreweave.com"
    python scripts/report/check_logged.py
"""
import wandb

api = wandb.Api()
P = "roisegel-personal/whale-upcall"

print("== runs in group resnet18_finetune_15ep ==")
for r in api.runs(P, filters={"group": "resnet18_finetune_15ep"}):
    h = r.history(keys=["epoch"], pandas=False)
    print(f"{r.name:45} state={r.state:9} epochs_logged={len(h)} created={r.created_at}")

print("\n== evaluation runs (PR curves / figures) ==")
for r in api.runs(P, filters={"jobType": "eval"}):
    keys = [k for k in r.summary.keys() if not k.startswith("_")]
    print(f"{r.name:25} created={r.created_at} keys={keys}")
