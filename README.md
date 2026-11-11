# Right whale upcall detection: spectrogram CNN vs. audio foundation model

Two detectors for North Atlantic right whale upcalls on the Kaggle
[ICML 2013 Whale Challenge – Right Whale Redux](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux)
data, compared on the same day-based cross-validation folds:

1. **Baseline** – log spectrogram + ImageNet ResNet-18, fine-tuned end to end.
2. **Foundation model** – frozen Perch v2 (ONNX) embeddings + a small probe.

**Status: work in progress.** The data pipeline, splits, Perch input selection
and baseline training loop are in place. Cross-validated results, the
evaluation code and the report are not finished yet.

## Data

47,841 labelled 2-second clips (11.0% contain an upcall) recorded on
28–31 March 2009, mono 16-bit at 2 kHz. The data is not in this repository;
download it from Kaggle and point `WHALE_DATA_DIR` at the folder that holds
`train2/` and `test2/`.

## Layout

```
src/whale/
  paths.py                 data and cache locations (env-overridable)
  data/manifest.py         filename -> date, 15-minute chunk, offset, label
  data/decode.py           AIFF -> one int16 array + parquet manifest
  data/preprocess.py       offset and level removal, edge fade, padding
  data/splits.py           four folds rotating over recording days
  data/audit.py            data checks and figures
  data/dataset.py          PyTorch dataset with time-shift augmentation
  features/spectrogram.py  log spectrogram, noise-floor removal, SpecAugment
  features/perch_input.py  2 kHz clip -> 5 s, 32 kHz Perch window
  models/resnet.py         ResNet-18 on spectrograms
  models/perch_onnx.py     Perch v2 ONNX wrapper
  embed.py                 resumable embedding extraction
  select_perch_input.py    input-variant selection on the design fold
  train.py                 training loop with W&B logging
  metrics.py               ranking metrics and operating points
configs/                   baseline training configs
reports/                   audit figures and selection results
tests/                     pytest suite
```

## Key decisions

- **Splits by recording day.** Neighbouring clips overlap in time and the
  positive rate swings from 0% to 37% by hour, so a random split leaks. Each of
  the four folds trains on two days, validates on one and tests on one. Fold 0
  is the design fold; its test day (31 March) is untouched by design choices.
- **Per-clip offset and level removal.** The recordings carry a constant offset
  and a level that differ between the labelled days and the Kaggle test days.
- **Perch input: 4× time compression.** Perch expects 5 s at 32 kHz and was
  trained mostly on sounds above the 80–250 Hz upcall band. Playing the clip
  4× faster moves the call into a range the model represents well. On the
  design fold's validation day this scored ROC-AUC 0.977 against 0.867 for the
  faithful-rate input (`reports/perch_input_selection.json`).

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. Everything runs on a
laptop CPU.

```bash
uv sync
uv run pytest
uv run python -m whale.data.decode train      # decode AIFF into the cache
uv run python -m whale.data.decode test
uv run python -m whale.data.audit             # data checks and figures
uv run python -m whale.select_perch_input     # Perch input-variant selection
uv run python -m whale.embed train --speed 4 --pad zero
uv run python -m whale.train --config configs/baseline.yaml --fold 0
```

Training logs to the W&B project `whale-upcall` when a key is configured and
runs offline otherwise.
