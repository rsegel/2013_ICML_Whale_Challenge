"""Stage 3 gate: can the laptop embed 48k clips with Perch, and what does
Perch actually see for each way of presenting a 2 kHz clip?

Prints the model's inputs/outputs and CPU throughput, and saves Perch's own
internal mel spectrogram for a few positive clips under each input variant.

Usage::

    python -m whale.models.perch_benchmark
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from whale import paths
from whale.data.preprocess import clean_clip
from whale.data.splits import load_labelled
from whale.features.perch_input import PerchInputConfig, make_perch_input
from whale.models.perch_onnx import PerchEmbedder

FIGURE_DIR = paths.REPO_ROOT / "reports" / "figures"
VARIANTS = [(1, "zero"), (2, "zero"), (4, "zero"), (8, "zero"), (4, "tile")]


def throughput(embedder: PerchEmbedder, batch: np.ndarray, repeats: int = 3) -> float:
    embedder.embed(batch)  # warm-up
    start = time.perf_counter()
    for _ in range(repeats):
        embedder.embed(batch)
    return repeats * len(batch) / (time.perf_counter() - start)


def main() -> None:
    manifest = load_labelled()
    waveforms = np.load(paths.waveform_path("train"), mmap_mode="r")
    train = manifest[manifest["split"] == "train"]
    rows = train.sample(32, random_state=0).index
    inputs = np.stack([make_perch_input(clean_clip(waveforms[r], manifest.at[r, "n_samples"])) for r in rows])

    for filename in ("perch_v2.onnx", "perch_v2_no_dft.onnx"):
        embedder = PerchEmbedder(filename)
        if filename == "perch_v2.onnx":
            for kind, items in embedder.describe().items():
                for item in items:
                    print(kind, item)
            zero = embedder.embed(np.zeros((1, 160_000), dtype=np.float32))
            print("embedding for a silent input:", zero.shape, "finite:", bool(np.isfinite(zero).all()))
        for batch_size in (1, 8, 32):
            rate = throughput(embedder, inputs[:batch_size])
            hours = len(manifest) / rate / 3600
            print(f"{filename} batch {batch_size:>2}: {rate:5.1f} clips/s -> {hours:.2f} h for {len(manifest)} labelled clips")

    # What Perch sees: its internal mel spectrogram (500 frames x 128 bins).
    embedder = PerchEmbedder("perch_v2.onnx")
    spectrogram_name = next(o.name for o in embedder.session.get_outputs() if o.shape[1:] == [500, 128])
    positives = train[(train["label"] == 1) & (train["n_samples"] == paths.CLIP_SAMPLES)].sample(3, random_state=3).index
    fig, axes = plt.subplots(len(positives), len(VARIANTS), figsize=(4 * len(VARIANTS), 3 * len(positives)))
    for i, row in enumerate(positives):
        clip = clean_clip(waveforms[row], manifest.at[row, "n_samples"])
        for j, (speed, pad) in enumerate(VARIANTS):
            wave = make_perch_input(clip, PerchInputConfig(speed, pad))[None]
            (mel,) = embedder.session.run([spectrogram_name], {embedder.input_name: wave})
            axes[i, j].imshow(mel[0].T, origin="lower", aspect="auto", cmap="magma")
            axes[i, j].set_title(f"speed x{speed}, {pad} pad", fontsize=9)
            axes[i, j].set_xlabel("Perch frame (5 s total)")
            axes[i, j].set_ylabel("Perch mel bin")
    fig.suptitle("Perch v2's internal mel spectrogram for three positive clips, by input variant")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "perch_input_variants.png", dpi=100)
    print("figure ->", FIGURE_DIR / "perch_input_variants.png")


if __name__ == "__main__":
    main()
