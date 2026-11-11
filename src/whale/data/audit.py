"""Data audit: print the facts later design choices rest on and save figures.

Reads the decoded cache (run ``python -m whale.data.decode`` first).

Usage::

    python -m whale.data.audit
"""

import hashlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

from whale import paths
from whale.data.preprocess import centre_pad

FIGURE_DIR = paths.REPO_ROOT / "reports" / "figures"
LISTEN_DIR = paths.REPO_ROOT / "reports" / "listen"
CLIP_SECONDS = paths.CLIP_SAMPLES / paths.SAMPLE_RATE


def load(subset: str) -> tuple[pd.DataFrame, np.ndarray]:
    manifest = pd.read_parquet(paths.manifest_path(subset))
    waveforms = np.load(paths.waveform_path(subset), mmap_mode="r")
    assert len(manifest) == len(waveforms)
    return manifest, waveforms


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def format_and_length(name: str, manifest: pd.DataFrame) -> None:
    section(f"{name}: format and length")
    print("clips:", len(manifest))
    print("sample rates:", manifest["sample_rate"].value_counts().to_dict())
    print("channel counts:", manifest["n_channels"].value_counts().to_dict())
    n = manifest["n_samples"]
    print(f"length == {paths.CLIP_SAMPLES}: {(n == paths.CLIP_SAMPLES).sum()}"
          f"  shorter: {(n < paths.CLIP_SAMPLES).sum()}  longer: {(n > paths.CLIP_SAMPLES).sum()}")
    print("length quantiles (samples):", n.quantile([0, 0.01, 0.05, 0.5, 1]).astype(int).to_dict())
    print("offset fraction digits seen:", sorted(manifest["offset_frac_raw"].unique().tolist()))


def class_balance(manifest: pd.DataFrame) -> None:
    section("train: class balance by day")
    table = manifest.groupby("date")["label"].agg(clips="size", positives="sum")
    table["pct_positive"] = (100 * table["positives"] / table["clips"]).round(1)
    table["chunks"] = manifest.groupby("date")["chunk_id"].nunique()
    print(table)
    print("overall pct positive:", round(100 * manifest["label"].mean(), 2))

    section("train: short clips by label")
    short = manifest["n_samples"] < paths.CLIP_SAMPLES
    print(pd.crosstab(short.rename("short"), manifest["label"]))
    print("pct positive among short clips:", round(100 * manifest.loc[short, "label"].mean(), 1))
    # How much could a model gain from clip length alone (visible as padding)?
    ranks = manifest["n_samples"].rank()
    n_pos = manifest["label"].sum()
    n_neg = len(manifest) - n_pos
    auc = (ranks[manifest["label"] == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    print("ROC-AUC of clip length as the only feature:", round(auc, 3))


def temporal_structure(name: str, manifest: pd.DataFrame) -> None:
    """How close in time are neighbouring clips? Decides whether a random
    split would put overlapping audio on both sides."""
    section(f"{name}: spacing of neighbouring clips inside a chunk")
    ordered = manifest.sort_values(["chunk_id", "offset_s"])
    gap = ordered.groupby("chunk_id")["offset_s"].diff().dropna()
    print("clips per chunk quantiles:",
          manifest.groupby("chunk_id").size().quantile([0, 0.25, 0.5, 0.75, 1]).astype(int).to_dict())
    print("gap between consecutive clip starts, quantiles (s):",
          gap.quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95]).round(1).to_dict())
    print(f"pct of gaps < {CLIP_SECONDS:.0f} s (clips overlap in time): {100 * (gap < CLIP_SECONDS).mean():.1f}")
    print(f"pct of gaps < 10 s: {100 * (gap < 10).mean():.1f}")
    # The offset counts from midnight, and the chunk is the 15-minute file it falls in.
    chunk_start = pd.to_timedelta(
        manifest["chunk"].str.replace(r"(..)(..)(..)", r"\1:\2:\3", regex=True)
    ).dt.total_seconds()
    within = manifest["offset_s"] - chunk_start
    print(f"offset minus chunk start: min {within.min():.1f} s, max {within.max():.1f} s (expect 0-900)")


def amplitude(name: str, manifest: pd.DataFrame, waveforms: np.ndarray) -> None:
    section(f"{name}: amplitude")
    data = np.asarray(waveforms, dtype=np.float32)
    peak = np.abs(data).max(axis=1)
    rms = np.sqrt((data**2).mean(axis=1))
    print("all-zero clips:", int((peak == 0).sum()))
    print("clips touching int16 full scale:", int((peak >= 32767).sum()))
    print("RMS quantiles:", np.quantile(rms, [0.01, 0.25, 0.5, 0.75, 0.99]).round(0).tolist())
    # Measured on the first 0.25 s, which is real audio in all but a handful of clips.
    head = data[:, :500]
    by_day = pd.DataFrame({"dc_offset": head.mean(axis=1), "std_after_dc_removal": head.std(axis=1)})
    print("median per day:\n", by_day.groupby(manifest["date"].values).median().round(0))
    if (manifest["label"] >= 0).all():
        for label in (0, 1):
            print(f"  median RMS label {label}:", round(float(np.median(rms[manifest['label'] == label])), 0))


def duplicates(train_wave: np.ndarray, test_wave: np.ndarray, train: pd.DataFrame) -> None:
    section("exact duplicate waveforms")
    digest = lambda row: hashlib.md5(np.ascontiguousarray(row).tobytes()).hexdigest()
    train_hash = pd.Series([digest(r) for r in train_wave])
    test_hash = pd.Series([digest(r) for r in test_wave])
    dup = train_hash.duplicated(keep=False)
    print("train clips sharing a waveform with another train clip:", int(dup.sum()))
    if dup.any():
        labels = train.loc[dup.values].assign(h=train_hash[dup].values).groupby("h")["label"].nunique()
        print("  duplicate groups with conflicting labels:", int((labels > 1).sum()))
    print("test clips duplicated inside test:", int(test_hash.duplicated(keep=False).sum()))
    print("test clips identical to a train clip:", int(test_hash.isin(set(train_hash)).sum()))


def spectrogram(clip: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs, times, power = signal.spectrogram(
        clip.astype(np.float32), fs=paths.SAMPLE_RATE, nperseg=256, noverlap=224, window="hann"
    )
    keep = freqs <= 500
    return freqs[keep], times, 10 * np.log10(power[keep] + 1e-6)


def figure_examples(manifest: pd.DataFrame, waveforms: np.ndarray, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for label, name in ((1, "positive"), (0, "negative")):
        rows = rng.choice(manifest.index[manifest["label"] == label], size=20, replace=False)
        fig, axes = plt.subplots(4, 5, figsize=(15, 9), sharex=True, sharey=True)
        for ax, row in zip(axes.ravel(), rows):
            real = np.asarray(waveforms[row][: manifest.at[row, "n_samples"]])
            clip = centre_pad(real, paths.CLIP_SAMPLES)
            freqs, times, db = spectrogram(clip)
            ax.pcolormesh(times, freqs, db, shading="auto", cmap="magma")
            ax.set_title(manifest.at[row, "filename"].removesuffix(".aif"), fontsize=7)
        fig.supxlabel("time (s)")
        fig.supylabel("frequency (Hz)")
        fig.suptitle(f"20 random {name} training clips (0-500 Hz, per-panel colour scale)")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"examples_{name}.png", dpi=110)
        plt.close(fig)


def figure_mean_spectrogram(manifest: pd.DataFrame, waveforms: np.ndarray) -> None:
    """Average log spectrogram per class: shows where in time and frequency the
    call sits, which sets the crop and the allowed time shift."""
    full = manifest.index[manifest["n_samples"] == paths.CLIP_SAMPLES]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    means = {}
    for label in (0, 1):
        rows = rng.choice(full[manifest.loc[full, "label"] == label], size=2000, replace=False)
        stack = [spectrogram(waveforms[r])[2] for r in rows]
        freqs, times, _ = spectrogram(waveforms[rows[0]])
        means[label] = np.mean(stack, axis=0)
    for ax, (data, title) in zip(axes, ((means[0], "mean negative"), (means[1], "mean positive"),
                                        (means[1] - means[0], "positive minus negative (dB)"))):
        mesh = ax.pcolormesh(times, freqs, data, shading="auto", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        fig.colorbar(mesh, ax=ax)
    axes[0].set_ylabel("frequency (Hz)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "mean_spectrogram_by_class.png", dpi=110)
    plt.close(fig)


def figure_rate_by_hour(manifest: pd.DataFrame) -> None:
    hour = manifest["chunk"].str[:2].astype(int)
    rate = manifest.assign(hour=hour).groupby(["date", "hour"])["label"].mean().unstack(0) * 100
    count = manifest.assign(hour=hour).groupby(["date", "hour"]).size().unstack(0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    rate.plot(ax=axes[0], marker="o")
    axes[0].set(title="Positive rate by hour of day", xlabel="hour", ylabel="% positive")
    count.plot(ax=axes[1], marker="o")
    axes[1].set(title="Clips by hour of day", xlabel="hour", ylabel="clips")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "positive_rate_by_hour.png", dpi=110)
    plt.close(fig)


def figure_lengths(manifest: pd.DataFrame) -> None:
    """Clip durations: all clips (left) and only those under 2 s (right)."""
    seconds = manifest["n_samples"] / paths.SAMPLE_RATE
    short = seconds[seconds < CLIP_SECONDS]
    edges = np.arange(0, CLIP_SECONDS + 0.125, 0.125)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    counts, _ = np.histogram(seconds, bins=np.append(edges, CLIP_SECONDS + 0.125))
    axes[0].bar(edges + 0.0625, counts, width=0.11, color="#2f6fb7")
    axes[0].annotate(f"{counts[-1]:,} clips are exactly 2.0 s\n({100 * counts[-1] / len(seconds):.1f}%)",
                     xy=(2.06, counts[-1]), xytext=(0.2, counts[-1] * 0.8),
                     arrowprops={"arrowstyle": "-", "color": "0.4"})
    axes[0].set(title=f"All {len(seconds):,} labelled clips", xlabel="clip duration (s)", ylabel="clips")

    axes[1].hist(short, bins=edges, color="#2f6fb7", rwidth=0.9)
    axes[1].set(title=f"The {len(short):,} clips shorter than 2 s ({100 * len(short) / len(seconds):.1f}%)",
                xlabel="clip duration (s)", ylabel="clips")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.9")
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "clip_durations.png", dpi=110)
    plt.close(fig)


def export_listening_clips(manifest: pd.DataFrame, waveforms: np.ndarray, per_class: int = 6) -> None:
    """Write a few clips as WAV, peak-normalised so they are audible."""
    LISTEN_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for label in (0, 1):
        for row in rng.choice(manifest.index[manifest["label"] == label], size=per_class, replace=False):
            clip = np.asarray(waveforms[row][: manifest.at[row, "n_samples"]], dtype=np.float32)
            clip = clip - clip.mean()
            clip = 0.9 * clip / max(np.abs(clip).max(), 1.0)
            name = manifest.at[row, "filename"].removesuffix(".aif")
            sf.write(LISTEN_DIR / f"label{label}_{name}.wav", clip, paths.SAMPLE_RATE)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    train, train_wave = load("train")
    test, test_wave = load("test")

    format_and_length("train", train)
    format_and_length("test", test)
    class_balance(train)
    temporal_structure("train", train)
    temporal_structure("test", test)
    amplitude("train", train, train_wave)
    amplitude("test", test, test_wave)
    duplicates(train_wave, test_wave, train)

    figure_examples(train, train_wave)
    figure_mean_spectrogram(train, train_wave)
    figure_rate_by_hour(train)
    figure_lengths(train)
    export_listening_clips(train, train_wave)
    print(f"\nfigures -> {FIGURE_DIR}\nlistening clips -> {LISTEN_DIR}")


if __name__ == "__main__":
    main()
