# Right whale upcall detection: spectrogram CNN vs. audio foundation model

Two detectors for North Atlantic right whale upcalls on the Kaggle
[ICML 2013 Whale Challenge – Right Whale Redux](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux)
data, compared on the same day-based cross-validation folds:

1. **Baseline**: log spectrogram + ImageNet ResNet-18, fine-tuned end to end.
2. **Foundation model**: frozen Perch v2 (ONNX) embeddings of a whitened, sped-up
   clip + a logistic-regression probe (a kNN and a one-hidden-layer network as
   controls).

The write-up (preprocessing, training, metrics, comparison) is in the
[W&B report](https://forge.coreweave.com/wandb/roisegel-personal/whale-upcall/reports/2013-ICML-Whale-Challenge-DeepVoice-Assignment-Nov-2026--VmlldzoxODMxMTY2Nw?accessToken=ni2v94gh18bf15umx4a7wvwvh5xw4vpxf68ythj4549iauy3jdgz3w94tiu1vv6w)
(hosted on CoreWeave Forge; the page can take up to a minute to load).

## Data

47,841 labelled 2-second clips recorded on 28–31 March 2009, mono 16-bit at
2 kHz. About 10% contain an upcall (measured on 28–29 March, the only days used
for label-based exploration). The Kaggle test clips (`test2/`, 4–6 April) are
unlabelled; they are decoded and checked for integrity but play no part in
model selection or evaluation. The data is not in this repository: download it
from Kaggle and point `WHALE_DATA_DIR` at the folder that holds `train2/` and
`test2/`. Caches (decoded audio, embeddings, checkpoints, predictions) go to
`WHALE_CACHE_DIR`, default `~/dv_cache`.

## Validation design

Neighbouring clips overlap in time and the positive rate swings from 0% to 37%
by hour, so a random split leaks. The four labelled days rotate through four
folds; each fold trains on two days, validates on one and tests on one:

| Fold | Train | Validation | Test |
|---|---|---|---|
| 0 (design fold) | 28, 29 Mar | 30 Mar | 31 Mar |
| 1 | 29, 30 Mar | 31 Mar | 28 Mar |
| 2 | 30, 31 Mar | 28 Mar | 29 Mar |
| 3 | 31, 28 Mar | 29 Mar | 30 Mar |

- Every design choice (Perch input variant, spectrogram settings,
  hyperparameters) is made on fold 0's validation day. Exploration that reads
  labels uses only fold 0's training days.
- The validation day picks the epoch (ResNet) or the regularisation strength
  (probes). The test day is only scored. Decision thresholds are set on the
  validation day and applied unchanged to the test day.
- Metrics are computed per fold and then averaged, never pooled across
  differently trained models. Every mean is reported with its per-fold values.
  Confidence intervals resample whole 15-minute chunks, paired across models.
- Main metric: average precision (upcalls are ~10% of clips). ROC-AUC
  (Kaggle's metric) and false alarms per 1,000 clips at fixed recall are also
  reported, the latter because a missed call costs more than a false alarm in
  a conservation setting.

## From raw audio to model input

- **Shared clean-up** (`data/preprocess.py`): per-clip offset and level
  removal (both differ between recording days), a short edge fade, and
  centred zero padding for the 11.6% of clips shorter than 2 s.
- **ResNet** (`features/spectrogram.py`): log spectrogram at the native 2 kHz
  rate, cropped to 40–500 Hz (upcalls sweep roughly 80–250 Hz), with each
  frequency's median subtracted to remove steady background noise. Training
  adds a random time shift (±0.25 s) and SpecAugment.
- **Perch** (`features/perch_input.py`): Perch expects 5 s at 32 kHz and was
  trained mostly on sounds far above the upcall band. Playing the clip 4×
  faster moves the call into a range the model represents well, and it is
  zero-padded to 5 s. On the design fold's validation day this scored
  ROC-AUC 0.977 against 0.867 for the faithful-rate input
  (`reports/perch_input_selection.json`). Restricting the pooling to the time
  frames that hold the clip was tested and gave no gain, so Perch's own pooled
  embedding is used.
- **Whitening the Perch input** (`--denoise whiten`): a waveform version of the
  ResNet's per-frequency background removal, applied before Perch. It raised
  mean validation AP from 0.895 to 0.919 (per fold +0.006 / +0.009 / +0.046 /
  +0.034) and is the final Perch model.

## Results (test days)

Each fold's model and decision threshold were frozen on its validation day; the
test day was scored once, at the end. Mean over the four test days with a 95%
chunk-bootstrap interval (`reports/test_2model/evaluation_test.json`).

| Average precision | 31 Mar | 28 Mar | 29 Mar | 30 Mar | Mean [95% interval] |
|---|---|---|---|---|---|
| ResNet-18, fine-tuned | 0.961 | 0.954 | 0.918 | 0.920 | 0.938 [0.930, 0.946] |
| Perch + whitening + logistic probe | 0.943 | 0.934 | 0.895 | 0.904 | 0.919 [0.908, 0.929] |

| False alarms per 1,000 clips at the validation day's 90%-recall threshold | 31 Mar | 28 Mar | 29 Mar | 30 Mar | Mean [95% interval] |
|---|---|---|---|---|---|
| ResNet-18, fine-tuned | 17.1 | 5.0 | 32.9 | 14.9 | 17.5 [15.7, 19.3] |
| Perch + whitening + logistic probe | 24.1 | 5.0 | 41.9 | 25.3 | 24.1 [21.9, 26.3] |

The ResNet is ahead on every test day (paired AP difference [+0.014, +0.026]).
With few labels the order flips: on fold 0, trained on 1% of the clips (227),
whitened Perch is ahead on all five subsamples (mean test AP 0.893 vs 0.836).

| | ResNet-18 fine-tune | Perch v2 + logistic probe |
|---|---|---|
| Trained parameters | 11.2 M (all) | 1,537 (101.8 M frozen) |
| Training per fold | 86-100 min on a laptop CPU; minutes on a GPU | under 1 minute on cached embeddings |
| Inference, laptop CPU | about 10 ms per clip | about 190 ms per clip |

## Layout

```
src/whale/
  paths.py                 data and cache locations (env-overridable)
  data/manifest.py         filename -> date, 15-minute chunk, offset, label
  data/decode.py           AIFF -> one int16 array + parquet manifest
  data/preprocess.py       offset and level removal, edge fade, padding
  data/splits.py           four folds rotating over recording days
  data/audit.py            integrity checks (all clips, no labels) and
                           label-based exploration (training days only)
  data/dataset.py          PyTorch dataset with time-shift augmentation
  features/spectrogram.py  log spectrogram, noise-floor removal, SpecAugment
  features/perch_input.py  2 kHz clip -> 5 s, 32 kHz Perch window
  models/resnet.py         ResNet-18 on spectrograms
  models/perch_onnx.py     Perch v2 ONNX wrapper (CPU or CUDA)
  models/perch_benchmark.py  Perch throughput and what its front end sees
  select_perch_input.py    input-variant selection on the design fold
  embed.py                 resumable embedding extraction (pooled or spatial)
  pool_spatial.py          centre-frame pooling of spatial embeddings
  train.py                 ResNet training loop, resumable, W&B logging
  probe.py                 logistic-regression or kNN probe on embeddings
  probe_mlp.py             one-hidden-layer probe, with training curves
  explore_embeddings.py    PCA and UMAP of embeddings vs spectrograms (saves the fit)
  analyse_errors.py        operating points and error gallery (validation days)
  evaluate.py              held-out metrics, confidence intervals, figures
  metrics.py               ranking metrics and operating points
  figure_label_efficiency.py         AP vs number of training clips (1%, 10%, 100%)
  figure_test_error_spectrograms.py  mean spectrogram of each model's test-day mistakes
  figure_test_error_maps.py          test-day mistakes projected onto the saved PCA/UMAP
  report_figures.py        data-exploration figures used in the report
  predict.py               score the unlabelled Kaggle clips (fold models averaged)
configs/                   ResNet configs (fine-tuned, frozen backbone)
scripts/run_heavy.ps1      every ResNet run in sequence, skip-and-resume
scripts/check_umap_repro.py  check that the UMAP maps reproduce (they do not across sessions)
scripts/report/            W&B report tooling: pull a report, build an edited draft, log figures
reports/                   result tables and figures (test_2model/: the reported comparison)
submissions/               Kaggle-format predictions
tests/                     pytest suite
```

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. Everything runs on a
CPU; a CUDA GPU speeds up ResNet training (picked automatically) and Perch
embedding (`--device cuda`, needs `onnxruntime-gpu`). The reported ResNet runs
and the spatial embeddings were computed on a GPU.

```bash
uv sync
uv run pytest

# data
uv run python -m whale.data.decode train      # decode AIFF into the cache
uv run python -m whale.data.decode test
uv run python -m whale.data.audit             # data checks and figures

# foundation model
uv run python -m whale.select_perch_input     # Perch input-variant selection
uv run python -m whale.embed train --speed 4 --pad zero
uv run python -m whale.embed train --speed 4 --pad zero --denoise whiten
uv run python -m whale.probe --fold 0         # repeat for folds 0-3
uv run python -m whale.probe --fold 0 --denoise whiten   # the final Perch model
uv run python -m whale.probe --fold 0 --model knn
uv run python -m whale.probe_mlp --fold 0

# baseline
uv run python -m whale.train --config configs/baseline.yaml --fold 0   # folds 0-3

# comparison
uv run python -m whale.evaluate               # validation days
uv run python -m whale.analyse_errors
uv run python -m whale.evaluate --split test --models resnet18_finetune perch_logreg_whiten
                                              # test days, once, at the end

# analysis figures (fold 0)
uv run python -m whale.explore_embeddings     # training days only; saves the map fit
uv run python -m whale.figure_test_error_spectrograms
uv run python -m whale.figure_test_error_maps
uv run python -m whale.figure_label_efficiency --split test --two-model

# Kaggle clips
uv run python -m whale.embed test --speed 4 --pad zero --denoise whiten
uv run python -m whale.predict perch
uv run python -m whale.predict resnet --checkpoints <folder with the four fold checkpoints>
```

Training runs resume from their last finished epoch when the same command is
run again. Runs log to the W&B project `whale-upcall` when a key is configured
and offline otherwise.
