"""Edit the user's working draft (15 Nov) into a NEW draft. Never saves over the draft.

DRY RUN by default: reads the draft (read-only), builds the edited copy in memory, writes
draft_edit_blocks.txt locally, uploads nothing. `--upload` saves a NEW draft with a new title.

    $env:WANDB_BASE_URL = "https://api.forge.coreweave.com"
    python scripts/report/log_results_figures.py      # once, before uploading: section-4 figure panels read that run
    python scripts/report/edit_draft.py               # dry run
    python scripts/report/edit_draft.py --upload      # create the new draft

Edits (15 Nov, user's request):
- Section 3 (Perch): every plot shows fold 0 only; per-fold tables replaced by a fold-0 line; setup text says
  the setting was chosen on fold 0. All per-fold tables move to one cross-validation check at the end of the
  section (they confirm the fold-0 choices, they did not make them). The four-fold MLP panel is dropped.
- UMAP/PCA panel restricted to run `embedding_figures` (the same images are also in run 24twkqfz, so the
  panel showed each twice).
- Section 2: the no-aug table (deleted from the draft by mistake) replaces its placeholder.
- Section 4: mistakes figures (fold-0 test day), cost table, label efficiency (fold-0 test day, 15 Nov).

Round 2 (15 Nov, after the round-1 upload; `--source <round-1 draft URL>` applies only these):
- Section 3 and 4 UMAP/PCA panels read run `embedding_maps_shared_fit` (one saved fit for both; section 4
  without the whitened-Perch row); mistakes intro and note updated to match.
- Discussion: pointer notes under "Model trade-offs" and "FP vs FN and tradeoff".
- Every panel's run table saved collapsed (patch below: the library drops active_runset on save).

Numbers: validation per-fold values as in the draft / NOTES.md (12-14 Nov); label efficiency NOTES.md 12-13 Nov;
cost NOTES.md "Practical trade-offs" (measured on the laptop CPU unless stated).
"""
import argparse
from pathlib import Path

import wandb_workspaces.reports.v2 as wr

DRAFT_URL = ("https://forge.coreweave.com/roisegel-personal/whale-upcall/reports/"
             "DeepVoice-Assignment-draft-15-11-13-20--VmlldzoxODMwODgzMA")
ENTITY, PROJECT = "roisegel-personal", "whale-upcall"
NEW_TITLE = "DeepVoice Assignment - draft 15-11, round 2 (collapsed run tables, discussion pointers)"
FOLD0 = 'Config("fold.value") == 0'
FULL_DATA = 'Config("train_fraction.value") == 1'


# ---------- locating blocks (each must match exactly one block, or the script stops) ----------
def find(blocks, cls, pred, what):
    hits = [i for i, b in enumerate(blocks) if isinstance(b, cls) and pred(b)]
    assert len(hits) == 1, f"{what}: expected 1 match, found {len(hits)}"
    return hits[0]


def text_starts(s):
    return lambda b: (getattr(b, "text", None) or "").startswith(s)


def grid_named(name):
    return lambda b: bool(b.runsets) and b.runsets[0].name == name


def replace(blocks, cls, pred, what, new):
    i = find(blocks, cls, pred, what)
    blocks[i:i + 1] = new if isinstance(new, list) else [new]


def edit_text(blocks, cls, start, old, new):
    """Replace one substring inside a text block, keeping the user's other wording."""
    i = find(blocks, cls, text_starts(start), start)
    assert old in blocks[i].text, f"{start!r}: substring {old!r} not found"
    blocks[i] = cls(text=blocks[i].text.replace(old, new))


# ---------- section 3: fold 0 only ----------
def fold0_setting_panel(name, group, setting, label):
    return wr.PanelGrid(
        runsets=[wr.Runset(entity=ENTITY, project=PROJECT, name=name,
                           filters=f"Metric(\"Group\") == '{group}' and {FULL_DATA} and {FOLD0}")],
        hide_run_sets=True,
        panels=[
            wr.LinePlot(x=setting, y=["val/average_precision"], log_x=True,
                        title=f"{label}, fold 0: validation AP vs {setting}", layout=wr.Layout(x=0, y=0, w=12, h=8)),
            wr.LinePlot(x=setting, y=["val/roc_auc"], log_x=True,
                        title=f"{label}, fold 0: validation ROC-AUC vs {setting}",
                        layout=wr.Layout(x=12, y=0, w=12, h=8)),
        ],
    )


def fold0_line(name, ap, auc, fa):
    return wr.MarkdownBlock(text=f"**{name}, fold 0 (validation day 30 Mar):** average precision {ap:.3f}, "
                                 f"ROC-AUC {auc:.3f}, false alarms per 1,000 clips at 90% recall {fa:.1f}.\n")


# per fold 0-3, validation days 30 / 31 / 28 / 29 Mar
VAL = {
    "Perch + logistic probe": {"ap": [0.894, 0.937, 0.889, 0.860], "auc": [0.981, 0.989, 0.974, 0.962],
                               "fa": [33.5, 23.1, 43.3, 49.9]},
    "Perch + kNN": {"ap": [0.831, 0.913, 0.843, 0.816], "auc": [0.957, 0.978, 0.956, 0.947],
                    "fa": [98.5, 36.3, 111.6, 120.3]},
    "Perch + MLP": {"ap": [0.894, 0.939, 0.891, 0.865], "auc": [0.980, 0.989, 0.977, 0.968],
                    "fa": [30.8, 23.8, 37.5, 47.8]},
    "Perch + whitening + logistic probe": {"ap": [0.900, 0.946, 0.935, 0.894], "auc": [0.986, 0.991, 0.990, 0.978],
                                           "fa": [26.4, 17.8, 11.6, 31.2]},
}
RESNET_VAL = {"ap": [0.927, 0.954, 0.956, 0.934], "auc": [0.991, 0.993, 0.996, 0.992], "fa": [19.3, 15.7, 9.0, 17.2]}


def section3_fold0(blocks):
    # logistic probe
    edit_text(blocks, wr.MarkdownBlock, "**Linear probe setup**",
              "chosen per fold by validation average precision. Best C = 0.01 on every fold.",
              "chosen by validation average precision on fold 0: C = 0.01.")
    replace(blocks, wr.PanelGrid, grid_named("Perch logistic probe"), "logistic panel",
            fold0_setting_panel("Perch logistic probe, fold 0", "perch_logreg", "C", "Logistic probe"))
    replace(blocks, wr.MarkdownBlock, text_starts("**Perch + logistic probe: validation days"), "logistic table",
            fold0_line("Perch + logistic probe", *(VAL["Perch + logistic probe"][k][0] for k in ("ap", "auc", "fa"))))
    # kNN
    edit_text(blocks, wr.MarkdownBlock, "**kNN setup**",
              "chosen per fold by validation average precision. Best k = 150 on every fold but",
              "chosen by validation average precision on fold 0: k = 150, the largest value tried, but AP")
    replace(blocks, wr.PanelGrid, grid_named("Perch kNN"), "kNN panel",
            fold0_setting_panel("Perch kNN, fold 0", "perch_knn", "k", "kNN"))
    replace(blocks, wr.MarkdownBlock, text_starts("**Perch + kNN: validation days"), "kNN table",
            fold0_line("Perch + kNN", *(VAL["Perch + kNN"][k][0] for k in ("ap", "auc", "fa"))))
    edit_text(blocks, wr.P, "Caveat: the best k was 150", "the largest value tried, on every fold, so",
              "the largest value tried, so")
    # MLP: the fold-0 40-epoch panel stays; the four-fold early-stopping panel (and the empty line above it) goes
    edit_text(blocks, wr.MarkdownBlock, "**MLP setup**", "best epoch per fold 7 / 2 / 10 / 5.",
              "best epoch on fold 0: 7.")
    i = find(blocks, wr.PanelGrid, grid_named("MLP, four folds, early stopping"), "four-fold MLP panel")
    assert isinstance(blocks[i - 1], wr.P) and not (blocks[i - 1].text or "").strip()
    del blocks[i - 1:i + 1]
    replace(blocks, wr.MarkdownBlock, text_starts("**Perch + MLP: validation days"), "MLP table",
            fold0_line("Perch + MLP", *(VAL["Perch + MLP"][k][0] for k in ("ap", "auc", "fa"))))
    # whitening
    edit_text(blocks, wr.MarkdownBlock, "**Whitening setup**", "Best C = 0.01 on every fold.",
              "Best C on fold 0 = 0.01.")
    replace(blocks, wr.PanelGrid, grid_named("Perch + whitening, logistic probe"), "whitening panel",
            fold0_setting_panel("Perch + whitening, logistic probe, fold 0", "perch_logreg_whiten", "C",
                                "Whitened logistic probe"))
    replace(blocks, wr.MarkdownBlock, text_starts("**Perch + whitening + logistic probe: validation days"),
            "whitening table", fold0_line("Perch + whitening + logistic probe",
                                          *(VAL["Perch + whitening + logistic probe"][k][0]
                                            for k in ("ap", "auc", "fa"))))


def cv_table(metric, label, fmt):
    head = ("| | Fold 0 (30 Mar) | Fold 1 (31 Mar) | Fold 2 (28 Mar) | Fold 3 (29 Mar) | Mean |\n"
            "|---|---|---|---|---|---|\n")

    def row(name, v):
        return f"| {name} | " + " | ".join(fmt.format(x) for x in v) + f" | {fmt.format(sum(v) / 4)} |\n"
    rows = "".join(row(n, d[metric]) for n, d in VAL.items())
    rows += row("*ResNet-18 fine-tuned (section 2, for reference)*", RESNET_VAL[metric])
    return f"*{label}*\n\n" + head + rows


def section3_cv_check(blocks):
    """After the last section-3 block, before '4. Results & Comparison'."""
    i = find(blocks, wr.H2, text_starts("4. Results"), "section 4 heading")
    intro = wr.P(text="All choices above were made on fold 0. To check that they hold on other recording days, "
                      "I re-ran every probe on all four folds. Each fold's setting is chosen on that same "
                      "validation day, so these numbers are slightly optimistic.")
    tables = wr.MarkdownBlock(text="**Validation days, per fold**\n\n"
                                   + cv_table("ap", "Average precision", "{:.3f}") + "\n"
                                   + cv_table("auc", "ROC-AUC", "{:.3f}") + "\n"
                                   + cv_table("fa", "False alarms per 1,000 clips at 90% recall", "{:.1f}"))
    settings = wr.MarkdownBlock(text=(
        "**Settings chosen on each fold** (folds 0 / 1 / 2 / 3)\n"
        "- Logistic probe: C = 0.01 / 0.01 / 0.01 / 0.01.\n"
        "- Whitened logistic probe: C = 0.01 / 0.01 / 0.01 / 0.01.\n"
        "- kNN: k = 150 / 150 / 150 / 150 (the largest value tried on every fold).\n"
        "- MLP: best epoch 7 / 2 / 10 / 5.\n"))
    facts = wr.P(text="On every fold kNN is the lowest of the Perch probes, the MLP is within 0.005 AP of the "
                      "logistic probe, and the whitened probe is the highest of the Perch probes. Fold 0 is "
                      "the fold where whitening helped least (AP +0.006; +0.046 on fold 2).")
    blocks[i:i] = [wr.H3(text="3.1 Cross-validation check of the fold-0 choices"), intro, tables, settings, facts]


def fix_embedding_panel(blocks):
    i = find(blocks, wr.PanelGrid, grid_named("embedding_figures"), "UMAP/PCA panel")
    old = blocks[i]
    blocks[i] = wr.PanelGrid(
        runsets=[wr.Runset(entity=ENTITY, project=PROJECT, name="embedding_figures",
                           filters="Metric(\"Name\") == 'embedding_figures'")],
        hide_run_sets=True, panels=old.panels)


# ---------- section 4 additions ----------
def results_panel(keys_heights):
    y, panels = 0, []
    for key, h in keys_heights:
        panels.append(wr.MediaBrowser(media_keys=[key], layout=wr.Layout(x=0, y=y, w=24, h=h)))
        y += h
    return wr.PanelGrid(runsets=[wr.Runset(entity=ENTITY, project=PROJECT, name="results_figures",
                                           filters="Metric(\"Name\") == 'results_figures'")],
                        hide_run_sets=True, panels=panels)


MISTAKES_INTRO = (
    "To see what the mistakes look like, I took fold 0's test day (31 Mar) and split each model's errors, at its "
    "validation-day 90%-recall threshold, into clips only whitened Perch gets wrong, clips only ResNet gets wrong, "
    "and clips both get wrong. The first figure is the mean spectrogram of each group (each clip whitened, full "
    "2-s clips only), with the clips both models get right for reference. The second places the same clips on "
    "UMAP maps fitted on the fold-0 training days, in each model's own input space (Perch embedding of the plain "
    "input, Perch embedding of the whitened input, and the log spectrogram the ResNet reads); the test clips are "
    "projected in, not refitted.")

MISTAKES_COUNTS = """\
**Fold 0 test day (31 Mar), clips per group** (each model at its own validation-day 90%-recall threshold)

| | Only whitened Perch wrong | Only ResNet wrong | Both wrong | Both right |
|---|---|---|---|---|
| False alarms (no call, flagged) | 166 | 69 | 172 | 11,902 |
| Missed calls (call, not flagged) | 68 | 73 | 58 | 1,546 |
"""

MISTAKES_NOTE = (
    "NOTE TO SELF, observations to write up (from NOTES.md, 14 Nov): every false-alarm group's mean shows an "
    "upcall-like upsweep at 100-200 Hz around 1 s, brightest for the ResNet-only false alarms, so some may be "
    "unlabelled or borderline calls (inferred; would need listening). Missed calls are faint (a few dB against "
    "about 8 dB for found calls); the both-missed mean barely shows a sweep and holds the most short clips "
    "(20 of 58). On the Perch UMAP most false alarms sit at the edge of the call lobe; a group of both-missed "
    "calls sits in a separate noise cluster. The PCA version (key test_errors_pca_fold0) is logged but not shown: "
    "spectrogram PCA is dominated by clip length. Delete this note when done.")

COST_TABLE = """\
**Cost** (measured on the laptop CPU unless stated)

| | ResNet-18 fine-tuned | Perch v2 + whitening + logistic probe |
|---|---|---|
| Parameters | 11.2 M, all trained | 101.8 M frozen + 1,537 trained |
| Model file | 45 MB | 409 MB (ONNX) + a few kB for the probe |
| One-time feature extraction | only the spectrogram (not timed separately) | Perch embedding of every clip, about 80 min for the labelled clips |
| Training, per fold | 86-100 min on the laptop CPU; 2.7 min for 15 epochs on a cluster GPU (fold 0) | under 1 min on the cached embeddings |
| Inference, per clip | about 10 ms (102 clips/s) | about 190 ms in batches of 32 (370 ms singly) |
| Scoring the 25,468 Kaggle clips, one fold model (computed from the per-clip rates) | about 4 min | about 81 min, almost all of it the embedding |
| Needs a GPU to iterate | in practice yes | no |

Timings for Perch were measured without whitening; whitening adds one preprocessing step per clip, not timed separately. \
The final submission averages the four fold models: four ResNet passes, but one Perch embedding pass plus four cheap probes.
"""

LABEL_EFF_INTRO = (
    "Fold 0 only, test day 31 Mar. Each model was retrained on a class-stratified 1% or 10% subsample of the "
    "fold-0 training clips, five subsamples (seeds 0-4) per fraction; seed s is the same subsample for both "
    "models, so the runs are paired. As for the full-data models, early stopping and C were set on the "
    "validation day (30 Mar); the test-day scores were saved at training time and only scored here (15 Nov).")


def label_eff_table():
    """Test-day AP per run, from reports/label_efficiency_test.csv (whale.figure_label_efficiency --split test)."""
    import pandas as pd
    runs = pd.read_csv(Path(__file__).resolve().parents[2] / "reports" / "label_efficiency_test.csv")
    ap = runs.pivot_table(index=["fraction", "seed"], columns="model", values="ap")
    out = ["**Label efficiency, fold 0 test day (31 Mar), average precision** "
           "(seeds 0 / 1 / 2 / 3 / 4, mean in brackets; 100% is one run)\n",
           "| Training clips | Perch + whitening + logistic probe | ResNet-18 fine-tuned | "
           "ResNet minus whitened Perch, per seed |", "|---|---|---|---|"]
    clips = {1.0: "22,692 (100%)", 0.1: "2,270 (10%)", 0.01: "227 (1%)"}

    def cell(v, fmt="{:.3f}"):
        return " / ".join(fmt.format(x) for x in v) + (f" ({fmt.format(v.mean())})" if len(v) > 1 else "")
    for frac in (1.0, 0.1, 0.01):
        part = ap.loc[frac]
        w, r = part["perch_logreg_whiten"], part["resnet18_finetune"]
        out.append(f"| {clips[frac]} | {cell(w)} | {cell(r)} | {cell(r - w, '{:+.3f}')} |")
    return "\n".join(out) + "\n"


LABEL_EFF_FACTS = (
    "At 1% the ResNet is below whitened Perch on all five subsamples, and its spread across subsamples is "
    "more than three times larger (0.776-0.877 against 0.878-0.906). At 10% it is ahead on four of five. The crossover "
    "lies between 227 and 2,270 training clips. The validation day (30 Mar) gives the same picture: ResNet behind "
    "on all five subsamples at 1% (mean AP 0.748 against 0.818) and ahead on four of five at 10% (0.882 against "
    "0.873). Fold 0 only.")


def section4_additions(blocks):
    """After the error-overlap table, before the training-dynamics note."""
    i = find(blocks, wr.MarkdownBlock, text_starts("**Which clips each model gets wrong**"), "overlap table") + 1
    blocks[i:i] = [
        wr.H3(text="4.1 What the mistakes look like (fold 0 test day)"),
        wr.P(text=MISTAKES_INTRO),
        wr.MarkdownBlock(text=MISTAKES_COUNTS),
        results_panel([("test_error_spectrograms_fold0", 12), ("test_errors_umap_fold0", 16)]),
        wr.P(text=MISTAKES_NOTE),
        wr.H3(text="4.2 Cost"),
        wr.MarkdownBlock(text=COST_TABLE),
        wr.H3(text="4.3 Label efficiency (fold 0 test day)"),
        wr.P(text=LABEL_EFF_INTRO),
        results_panel([("label_efficiency_test_2model", 12)]),
        wr.MarkdownBlock(text=label_eff_table()),
        wr.P(text=LABEL_EFF_FACTS),
    ]


AUG_TABLE = """\
**Augmentation ablation: validation average precision per fold** (seed 0, one run per arm)

| | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Mean |
|---|---|---|---|---|---|
| ResNet + aug | 0.927 | 0.954 | 0.956 | 0.934 | 0.943 |
| ResNet, no aug | 0.927 | 0.958 | 0.952 | 0.931 | 0.942 |
| Difference (aug - no aug) | -0.000 | -0.003 | +0.004 | +0.002 | +0.001 |
"""


def section2_aug_table(blocks):
    """The no-aug table was deleted from the draft by mistake; it replaces the placeholder."""
    replace(blocks, wr.P, text_starts("**insert no-aug table here**"), "no-aug placeholder",
            wr.MarkdownBlock(text=AUG_TABLE))


# ---------- round 2 (15 Nov, after the first upload) ----------
# wandb-workspaces never writes PanelGrid.active_runset into the saved spec, so every grid is saved with its
# run table open (open_run_set = 0). This passes it through; None = collapsed. hide_run_sets only hides the
# tables from viewers, not from the author in edit mode.
_orig_to_model = wr.PanelGrid._to_model


def _to_model_keep_runset_state(self):
    model = _orig_to_model(self)
    model.metadata.open_run_set = self.active_runset
    return model


wr.PanelGrid._to_model = _to_model_keep_runset_state


def collapse_run_tables(blocks):
    for i, b in enumerate(blocks):
        if isinstance(b, wr.PanelGrid):
            b = b.model_copy() if hasattr(b, "model_copy") else b
            b.active_runset = None
            b.hide_run_sets = True
            blocks[i] = b


TRADEOFF_POINTERS = """\
NOTE TO SELF, discussion pointers on model trade-offs (all numbers already in the report; delete when written):
- ResNet is ahead on every test day: mean AP 0.938 vs 0.919, paired difference [+0.014, +0.026], +0.016 to +0.023 on each day; false alarms at the 90%-recall threshold 17.5 vs 24.1 per 1,000 clips, paired [-7.8, -5.4].
- Perch is cheap to adapt: 1,537 trained parameters and under 1 min per fold on cached embeddings, against 86-100 min per fold on the laptop CPU (2.7 min on a GPU) for the ResNet. The cost moves to inference: about 190 ms vs 10 ms per clip, 81 vs 4 min for the Kaggle clips. Both are fast enough for a live 2-s detector.
- Label efficiency (4.3): with 227 labelled clips whitened Perch is ahead on all five subsamples and far more stable; from 2,270 the ResNet is ahead on four of five. Possible decision rule: few labels, many call types or no GPU, probe a foundation model; many labels for one call type, fine-tune a small CNN.
- The comparison is fine-tuned vs frozen by design; whitening the Perch input closed about half the validation gap (mean AP 0.895 to 0.919, ResNet 0.943). Perch was never fine-tuned.
"""

FPFN_POINTERS = """\
NOTE TO SELF, discussion pointers on false alarms vs missed calls (delete when written):
- Which error costs more depends on the use: for a ship-strike alert a missed call is the expensive one, so recall is fixed (90%) and false alarms are the cost. In workload terms, on 31 Mar (14,054 clips) 24.1 vs 17.1 per 1,000 is about 340 vs 240 clips a day for a person to check.
- The threshold set on the validation day does not fully carry over: mean recall reached on the test days is 0.884 (Perch) and 0.880 (ResNet) against the 0.90 target, and only 0.79-0.81 on 28 Mar for both. A deployed detector would need recalibration per site or season, or a safety margin.
- The two models make different mistakes: over the four test days 581 false alarms are Perch's alone, 239 ResNet's alone, 679 shared; missed calls 240 / 263 / 278. About half of each model's misses are its own, so averaging the two scores is a natural next step (not tested).
- 4.1: false-alarm means show a call-like upsweep, so some may be unlabelled calls (would need an expert to listen); missed calls are faint, and the both-missed group holds the most short clips (20 of 58).
- Limits to state: one seed per full-data configuration (five for label efficiency, fold 0 only); four days at one site; validation-day numbers are optimistic because settings were picked on them. Next steps: score averaging, more seeds, a fine-tunable audio model (e.g. BEATs), a background estimate from neighbouring clips instead of the clip itself (short clips).
"""


def discussion_pointers(blocks):
    for start, text in (("Model trade-offs", TRADEOFF_POINTERS), ("FP vs FN and tradeoff", FPFN_POINTERS)):
        i = find(blocks, wr.P, lambda b, s=start: (b.text or "").strip() == s, start)
        blocks[i + 1:i + 1] = [wr.MarkdownBlock(text=text)]


SHARED_RUN = "embedding_maps_shared_fit"  # logged by log_shared_fit_figures.py


def shared_panel(keys_heights):
    y, panels = 0, []
    for key, h in keys_heights:
        panels.append(wr.MediaBrowser(media_keys=[key], layout=wr.Layout(x=0, y=y, w=24, h=h)))
        y += h
    return wr.PanelGrid(runsets=[wr.Runset(entity=ENTITY, project=PROJECT, name=SHARED_RUN,
                                           filters=f"Metric(\"Name\") == '{SHARED_RUN}'")],
                        hide_run_sets=True, panels=panels)


MISTAKES_INTRO_2 = (
    "To see what the mistakes look like, I took fold 0's test day (31 Mar) and split each model's errors, at its "
    "validation-day 90%-recall threshold, into clips only whitened Perch gets wrong, clips only ResNet gets wrong, "
    "and clips both get wrong. The first figure is the mean spectrogram of each group (each clip whitened, full "
    "2-s clips only), with the clips both models get right for reference. The second places the same clips on the "
    "UMAP maps of section 3 (the same fit, not refitted; the test clips are projected in): the Perch embedding of "
    "the plain input, and the log spectrogram the ResNet reads. Whitened Perch reads the embedding of the whitened "
    "input, so the top row shows where its mistakes fall in the Perch map of section 3, not in its own input.")

MISTAKES_NOTE_2 = (
    "NOTE TO SELF, observations to write up (from NOTES.md, 14-15 Nov; map readings are mine): every false-alarm "
    "group's mean shows an upcall-like upsweep at 100-200 Hz around 1 s, brightest for the ResNet-only false alarms, "
    "so some may be unlabelled or borderline calls (inferred; would need listening). Missed calls are faint (a few "
    "dB against about 8 dB for found calls); the both-missed mean barely shows a sweep and holds the most short "
    "clips (20 of 58). Perch map: false alarms sit at the neck between the noise mass and the upcall lobe and "
    "inside the lobe; missed calls mostly along the lobe, with a few both-missed calls out in the small noise "
    "islands on the left, far from the calls. Spectrogram map: errors sit along the edge between the noise and "
    "upcall regions, and the separate arm (short clips, by the clip-length panel of section 3) holds many of "
    "the both-missed calls. The PCA version (key test_errors_pca_fold0) is logged but not shown: spectrogram PCA "
    "is dominated by clip length. Delete this note when done.")


def shared_fit_maps(blocks):
    """Section 3 and 4 maps from one saved fit (run embedding_maps_shared_fit); section 4 without the whitened row."""
    replace(blocks, wr.PanelGrid, grid_named("embedding_figures"), "section-3 UMAP/PCA panel",
            shared_panel([("embedding_structure_umap", 12), ("embedding_structure_pca", 12)]))
    i = find(blocks, wr.PanelGrid,
             lambda b: any("test_errors_umap_fold0" in (getattr(p, "media_keys", None) or []) for p in b.panels),
             "section-4 mistakes panel")
    blocks[i] = shared_panel([("test_error_spectrograms_fold0", 12), ("test_errors_umap_fold0", 12)])
    replace(blocks, wr.P, text_starts("To see what the mistakes look like"), "mistakes intro",
            wr.P(text=MISTAKES_INTRO_2))
    replace(blocks, wr.P, text_starts("NOTE TO SELF, observations to write up"), "mistakes note",
            wr.P(text=MISTAKES_NOTE_2))


ROUND1 = [section2_aug_table, section3_fold0, section3_cv_check, fix_embedding_panel, section4_additions]
ROUND2 = [shared_fit_maps, discussion_pointers, collapse_run_tables]

ap = argparse.ArgumentParser()
ap.add_argument("--upload", action="store_true")
ap.add_argument("--source", default=None,
                help="URL of the draft uploaded in round 1; only the round-2 edits are applied to it. Without it "
                     "the 15-11 draft is rebuilt with round 1 + round 2.")
args = ap.parse_args()
upload = args.upload

src_url = (args.source.replace("/wandb/", "/") if args.source else DRAFT_URL)
src = wr.Report.from_url(src_url)  # read-only
blocks = list(src.blocks)  # new list; replaced blocks are new objects, the loaded draft is not modified
for edit in (ROUND2 if args.source else ROUND1 + ROUND2):
    edit(blocks)

new = wr.Report(entity=ENTITY, project=PROJECT, title=NEW_TITLE,
                description="Edited copy of an earlier draft. That draft is untouched.", blocks=blocks)

with open("draft_edit_blocks.txt", "w", encoding="utf-8") as f:
    for i, b in enumerate(new.blocks):
        f.write(f"--- [{i}] {type(b).__name__} ---\n{b}\n\n")
print(f"Dry run: {len(src.blocks)} -> {len(new.blocks)} blocks; wrote draft_edit_blocks.txt")

if upload:
    new.save(draft=True)  # creates a NEW draft report; the 15-11 draft is not touched
    print("Uploaded draft:", new.url)
else:
    print("Not uploaded. Run log_shared_fit_figures.py first, then re-run with --upload (and --source <round-1 draft URL>).")
