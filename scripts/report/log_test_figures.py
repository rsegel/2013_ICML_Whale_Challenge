"""Log the existing test-day evaluation figures to W&B as images (one new run).

Run by the user; it creates one new run in whale-upcall and touches nothing else.
The PNGs already exist in reports/test_2model (from `whale.evaluate --split test --models resnet18_finetune
perch_logreg_whiten`, whitened Perch vs ResNet only); nothing is recomputed.

    $env:WANDB_BASE_URL = "https://api.forge.coreweave.com"
    python scripts/report/log_test_figures.py
"""
from pathlib import Path

import wandb

FIG = Path(__file__).resolve().parents[2] / "reports" / "test_2model"
KEYS = {"evaluation_test_pr_calibration_2model": FIG / "evaluation_test.png",
        "evaluation_test_per_day_2model": FIG / "evaluation_test_folds.png"}
for p in KEYS.values():
    assert p.exists(), p

run = wandb.init(project="whale-upcall", entity="roisegel-personal", name="test_figures_2model", job_type="eval",
                 config={"split": "test", "models": ["perch_logreg_whiten", "resnet18_finetune"],
                         "n_boot": 1000})
run.log({k: wandb.Image(str(p)) for k, p in KEYS.items()})
run.finish()
print("logged:", ", ".join(KEYS))
