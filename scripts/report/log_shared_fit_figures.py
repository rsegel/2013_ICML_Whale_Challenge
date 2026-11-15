"""Log the section-3 and section-4 map figures, now drawn from ONE saved UMAP/PCA fit (one new run).

Run by the user; it creates one new run in whale-upcall (`embedding_maps_shared_fit`) and touches nothing else.
Why: on 12 Nov the section-3 maps were fitted once, and on 14-15 Nov the section-4 error maps refitted them;
same data, code and seed, different UMAP layout. Since 15 Nov `whale.explore_embeddings` saves its fit
(dv_cache/embedding_maps_fold0.joblib) and `whale.figure_test_error_maps` projects into that saved fit, so both
figures below share one map. Section 4 shows plain-input Perch + log spectrogram only (whitened row dropped).
The older runs (`embedding_figures`, `results_figures`) keep the previous versions; the report panels are
pointed at this run by edit_draft.py (round 2).

    $env:WANDB_BASE_URL = "https://api.forge.coreweave.com"
    python scripts/report/log_shared_fit_figures.py
"""
from pathlib import Path

import wandb

FIG = Path(__file__).resolve().parents[2] / "reports" / "figures"
KEYS = {"embedding_structure_umap": FIG / "embedding_structure_umap.png",
        "embedding_structure_pca": FIG / "embedding_structure_pca.png",
        "test_errors_umap_fold0": FIG / "test_errors_umap_fold0.png",
        "test_errors_pca_fold0": FIG / "test_errors_pca_fold0.png",
        "test_error_spectrograms_fold0": FIG / "test_error_spectrograms_fold0.png"}
for p in KEYS.values():
    assert p.exists(), p

run = wandb.init(project="whale-upcall", entity="roisegel-personal", name="embedding_maps_shared_fit",
                 job_type="eval",
                 config={"fit": "fold-0 training days 28-29 Mar, 3,000 clips per day, saved 15 Nov",
                         "test_projection": "fold 0 test day 31 Mar, projected into the saved fit"})
run.log({k: wandb.Image(str(p)) for k, p in KEYS.items()})
run.finish()
print("logged:", ", ".join(KEYS))
