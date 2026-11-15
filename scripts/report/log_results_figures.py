"""Log the existing section-4 figures to W&B as images (one new run, `results_figures`).

Run by the user; it creates one new run in whale-upcall and touches nothing else. Nothing is recomputed:
the PNGs already exist in reports/figures.
- test_error_spectrograms_fold0 / test_errors_umap_fold0 / test_errors_pca_fold0: fold-0 test day (31 Mar)
  mistakes of whitened Perch and ResNet (whale.figure_test_error_spectrograms / figure_test_error_maps, 14 Nov).
- label_efficiency_test_2model: fold 0 TEST day (31 Mar), whitened Perch vs ResNet, 5 seeds per model at 1% and
  10% (whale.figure_label_efficiency --split test --two-model, 15 Nov). Shown in section 4.
- label_efficiency_val: the same runs on the validation day (30 Mar), three models. Logged for reference only.

    $env:WANDB_BASE_URL = "https://api.forge.coreweave.com"
    python scripts/report/log_results_figures.py
"""
from pathlib import Path

import wandb

FIG = Path(__file__).resolve().parents[2] / "reports" / "figures"
KEYS = {"test_error_spectrograms_fold0": FIG / "test_error_spectrograms_fold0.png",
        "test_errors_umap_fold0": FIG / "test_errors_umap_fold0.png",
        "test_errors_pca_fold0": FIG / "test_errors_pca_fold0.png",
        "label_efficiency_test_2model": FIG / "label_efficiency_test_2model.png",
        "label_efficiency_val": FIG / "label_efficiency_val.png"}
for p in KEYS.values():
    assert p.exists(), p

run = wandb.init(project="whale-upcall", entity="roisegel-personal", name="results_figures", job_type="eval",
                 config={"error_figures": "fold 0 test day, whitened Perch vs ResNet",
                         "label_efficiency": "fold 0 test day (section 4) and validation day"})
run.log({k: wandb.Image(str(p)) for k, p in KEYS.items()})
run.finish()
print("logged:", ", ".join(KEYS))
