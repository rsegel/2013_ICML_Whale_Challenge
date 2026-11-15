"""Log the existing PCA / UMAP embedding figures to W&B as images (one new run).

Run by the user; it creates one new run in whale-upcall and touches nothing else.
The PNGs already exist in reports/figures; nothing is recomputed.

    $env:WANDB_BASE_URL = "https://api.forge.coreweave.com"
    python scripts/report/log_embedding_figures.py
"""
from pathlib import Path

import wandb

FIG = Path(__file__).resolve().parents[2] / "reports" / "figures"
KEYS = {"embedding_structure_umap": FIG / "embedding_structure_umap.png",
        "embedding_structure_pca": FIG / "embedding_structure_pca.png"}
for p in KEYS.values():
    assert p.exists(), p

run = wandb.init(project="whale-upcall", entity="roisegel-personal", name="embedding_figures", job_type="data",
                 config={"days": ["20090328", "20090329"], "clips_per_day": 3000, "fold": 0})
run.log({k: wandb.Image(str(p)) for k, p in KEYS.items()})
run.finish()
print("logged:", ", ".join(KEYS))
