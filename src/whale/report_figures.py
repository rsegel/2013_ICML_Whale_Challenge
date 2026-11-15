"""Three report figures that justify the data handling, on 28-29 March only.

1. ``why_split_by_day``: neighbouring clips overlap in time and labels persist.
2. ``per_clip_normalisation``: raw offset and level per clip, and what cleaning does.
3. ``padding_example``: a short clip before and after fade and centre padding.
4. ``spectrogram_pipeline``: one training-day upcall as a raw spectrogram, whitened,
   time-shifted, SpecAugment-masked and both augmentations together.

Only the design fold's training days (28 and 29 March) are read, so no
validation or test day is touched. Reads the decoded cache.

Usage::

    python -m whale.report_figures            # write the PNGs
    python -m whale.report_figures --wandb    # also log them to W&B
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from whale import paths
from whale.data.preprocess import centre_pad, clean_clip, fade_edges
from whale.data.splits import DESIGN_FOLD, FOLDS, load_labelled
from whale.features.spectrogram import LogSpectrogram, spec_augment

FIGURE_DIR = paths.REPO_ROOT / "reports" / "figures"
CLIP_SECONDS = paths.CLIP_SAMPLES / paths.SAMPLE_RATE
BLUE, ORANGE = "#2f6fb7", "#e08a1e"
DAY_COLOURS = {0: BLUE, 1: ORANGE}


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9")
    ax.set_axisbelow(True)


def previous_clip_stats(explore: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Per day: P(positive | previous clip in the same chunk is positive) vs base rate,
    and the gap in seconds between consecutive clip starts inside a chunk."""
    ordered = explore.sort_values(["chunk_id", "offset_s"])
    previous = ordered.groupby("chunk_id")["label"].shift()
    gap = ordered.groupby("chunk_id")["offset_s"].diff()
    after_pos = ordered[previous == 1]
    rows = []
    for day, part in ordered.groupby("date"):
        follow = after_pos[after_pos["date"] == day]
        rows.append({"day": day, "base_rate": 100 * part["label"].mean(),
                     "after_positive": 100 * follow["label"].mean(),
                     "n_after_positive": len(follow), "n_clips": len(part)})
    return pd.DataFrame(rows).set_index("day"), gap.dropna()


def figure_split_by_day(explore: pd.DataFrame) -> None:
    table, gap = previous_clip_stats(explore)
    print("\nlabel persistence (training days):\n", table.round(1).to_string())
    print(f"gaps < {CLIP_SECONDS:.0f} s: {100 * (gap < CLIP_SECONDS).mean():.1f}%  (n={len(gap):,})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    edges = np.arange(0, 20.5, 0.5)
    axes[0].hist(gap.clip(upper=20), bins=edges, color=BLUE, rwidth=0.9)
    axes[0].axvline(CLIP_SECONDS, color="0.3", ls="--")
    axes[0].text(CLIP_SECONDS + 5.5, axes[0].get_ylim()[1] * 0.9,
                 f"clip length 2 s:\n{100 * (gap < CLIP_SECONDS).mean():.0f}% of gaps are shorter,\n"
                 "so those clips overlap in time", va="top", fontsize=9)
    axes[0].set(title="Gap between consecutive clip starts (within a 15-min chunk)",
                xlabel="seconds (20 = 20 or more)", ylabel="clip pairs")

    x = np.arange(len(table))
    w = 0.36
    axes[1].bar(x - w / 2, table["base_rate"], w, color="0.7", label="any clip (base rate)")
    axes[1].bar(x + w / 2, table["after_positive"], w, color=ORANGE, label="clip right after a positive")
    for i, (_, row) in enumerate(table.iterrows()):
        axes[1].text(i - w / 2, row["base_rate"] + 0.5, f"{row['base_rate']:.1f}%", ha="center", fontsize=9)
        axes[1].text(i + w / 2, row["after_positive"] + 0.5,
                     f"{row['after_positive']:.1f}%\n(n={int(row['n_after_positive']):,})", ha="center", fontsize=9)
    axes[1].set_xticks(x, [f"{d[-2:]} Mar" for d in table.index])
    axes[1].set(title="Labels persist in time: % positive", ylabel="% positive",
                ylim=(0, table["after_positive"].max() * 1.35))
    axes[1].legend(frameon=False, loc="upper right")
    for ax in axes:
        style(ax)
    fig.suptitle("Why the split is by day: clips are not independent (28-29 March, training days of the design fold)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "why_split_by_day.png", dpi=130)
    plt.close(fig)


def figure_normalisation(explore: pd.DataFrame, waveforms: np.ndarray) -> None:
    """Raw offset and level of each clip's first 0.25 s (real audio in nearly all clips)."""
    rows = explore.index.to_numpy()
    head = np.asarray(waveforms[rows][:, :500], dtype=np.float32)
    offset, level = head.mean(axis=1), head.std(axis=1)
    cleaned = np.stack([clean_clip(waveforms[r], explore.at[r, "n_samples"])[:500] for r in rows[:2000]])
    print("\nraw offset and level of the first 0.25 s, median per day:")
    print(pd.DataFrame({"offset": offset, "level_std": level}).groupby(explore["date"].values).median().round(0))
    print(f"after cleaning (first 2,000 clips): mean {cleaned.mean(axis=1).mean():.3f}, "
          f"std {cleaned.std(axis=1).mean():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, values, title, xlabel in (
        (axes[0], offset, "Raw offset (mean of the clip)", "counts"),
        (axes[1], level, "Raw level (std after removing the offset)", "counts"),
    ):
        lo, hi = np.quantile(values, [0.005, 0.995])
        bins = np.linspace(lo, hi, 60)
        for i, day in enumerate(sorted(explore["date"].unique())):
            ax.hist(values[(explore["date"] == day).to_numpy()], bins=bins, alpha=0.6,
                    color=DAY_COLOURS[i], label=f"{day[-2:]} Mar, median {np.median(values[(explore['date'] == day).to_numpy()]):.0f}")
        ax.set(title=title, xlabel=xlabel if False else "value (int16 counts)", ylabel="clips")
        ax.legend(frameon=False)
        style(ax)
    fig.suptitle("Offset and level vary from clip to clip, so each clip is standardised on its own "
                 "(afterwards every clip has mean 0, std 1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "per_clip_normalisation.png", dpi=130)
    plt.close(fig)


def figure_padding(explore: pd.DataFrame, waveforms: np.ndarray) -> None:
    """A short clip and a full clip, as the models see them (clean, fade, centre-pad)."""
    n = explore["n_samples"].to_numpy()
    short_row = explore.index[np.argmin(np.abs(n - 1.4 * paths.SAMPLE_RATE) + (n >= paths.CLIP_SAMPLES) * 1e9)]
    full_row = explore.index[(explore["n_samples"] == paths.CLIP_SAMPLES).to_numpy()][0]
    t = np.arange(paths.CLIP_SAMPLES) / paths.SAMPLE_RATE

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    for ax, row, name in ((axes[0], full_row, "Full clip"), (axes[1], short_row, "Short clip")):
        length = int(explore.at[row, "n_samples"])
        padded = centre_pad(fade_edges(clean_clip(waveforms[row], length)), paths.CLIP_SAMPLES)
        ax.plot(t, padded, lw=0.7, color=BLUE)
        start = (paths.CLIP_SAMPLES - length) // 2
        if length < paths.CLIP_SAMPLES:
            ax.axvspan(0, start / paths.SAMPLE_RATE, color="0.85")
            ax.axvspan((start + length) / paths.SAMPLE_RATE, CLIP_SECONDS, color="0.85")
            ax.text(CLIP_SECONDS / 2, ax.get_ylim()[1] * 0.85, "real audio", ha="center", fontsize=9)
            ax.text(start / paths.SAMPLE_RATE / 2, ax.get_ylim()[1] * 0.85, "zeros", ha="center", fontsize=9)
            ax.text(CLIP_SECONDS - start / paths.SAMPLE_RATE / 2, ax.get_ylim()[1] * 0.85, "zeros", ha="center", fontsize=9)
        ax.set(title=f"{name}: {length / paths.SAMPLE_RATE:.2f} s of audio", ylabel="standardised amplitude")
        style(ax)
    axes[1].set_xlabel("seconds in the 2 s model input")
    fig.suptitle("Short clips are standardised, faded at the edges and centred in zeros to reach 2 s "
                 "(a training-day example)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "padding_example.png", dpi=130)
    plt.close(fig)


def figure_spectrogram_pipeline(explore: pd.DataFrame, waveforms: np.ndarray, seed: int = 3) -> None:
    """One row: raw log spectrogram -> whitened (the CNN input) -> each augmentation.

    The clip is a random full-length positive from 28-29 March. "Whitened" is the
    CNN's step of subtracting each frequency's median dB level over time (the
    same as dividing by the median magnitude), then clamping to [-10, +30] dB.
    """
    import torch

    rng = np.random.default_rng(seed)
    full_pos = explore.index[((explore["n_samples"] == paths.CLIP_SAMPLES) & (explore["label"] == 1)).to_numpy()]
    row = int(rng.choice(full_pos))
    clip = fade_edges(clean_clip(waveforms[row], paths.CLIP_SAMPLES))

    def place(shift: int) -> np.ndarray:
        out = np.zeros(paths.CLIP_SAMPLES, dtype=np.float32)
        lo, hi = max(shift, 0), min(shift + len(clip), paths.CLIP_SAMPLES)
        out[lo:hi] = clip[lo - shift : hi - shift]
        return out

    shift = 400  # 0.2 s; training draws uniformly from +-500 samples
    spec_layer = LogSpectrogram()

    def whitened(wave: np.ndarray) -> torch.Tensor:
        return spec_layer(torch.from_numpy(wave)[None])

    def raw_db(wave: np.ndarray) -> np.ndarray:
        stft = torch.stft(torch.from_numpy(wave)[None], spec_layer.n_fft, spec_layer.hop_length,
                          spec_layer.win_length, spec_layer.window, center=True, pad_mode="constant",
                          return_complex=True)
        power = stft[:, spec_layer.first_bin : spec_layer.last_bin].abs().square()
        return (10 * torch.log10(power + 1e-6))[0].numpy()

    torch.manual_seed(seed)
    base = whitened(place(0))
    masked = spec_augment(base)
    shifted = whitened(place(shift))
    torch.manual_seed(seed)
    both = spec_augment(shifted)
    panels = [
        ("1. Raw log spectrogram", raw_db(place(0)), "dB"),
        ("2. Whitened (CNN input)", base[0, 0].numpy(), "scaled, clamped"),
        (f"3. Whitened + time shift\n(+{shift / paths.SAMPLE_RATE:.1f} s here; up to +-0.25 s in training)",
         shifted[0, 0].numpy(), "scaled, clamped"),
        ("4. Whitened + SpecAugment\n(one frequency band, one time band)", masked[0, 0].numpy(), "scaled, clamped"),
        ("5. Both, as in training", both[0, 0].numpy(), "scaled, clamped"),
    ]
    f_lo = spec_layer.first_bin * paths.SAMPLE_RATE / spec_layer.n_fft
    f_hi = spec_layer.last_bin * paths.SAMPLE_RATE / spec_layer.n_fft
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.9), sharey=True)
    for i, (ax, (title, image, unit)) in enumerate(zip(axes, panels)):
        shown = ax.imshow(image, origin="lower", aspect="auto", cmap="magma",
                          extent=(0, CLIP_SECONDS, f_lo, f_hi))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("seconds")
        fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.03, label=unit)
    axes[0].set_ylabel("frequency (Hz)")
    fig.suptitle(f"Same training-day upcall through the CNN's input pipeline "
                 f"(28-29 March clip {explore.at[row, 'filename']}; offset and level removed first)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "spectrogram_pipeline.png", dpi=130)
    plt.close(fig)


def main(log_wandb: bool) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_labelled(DESIGN_FOLD)
    waveforms = np.load(paths.waveform_path("train"), mmap_mode="r")
    explore = manifest[manifest["split"] == "train"]  # keeps the row index into the waveforms
    assert set(explore["date"]) == set(FOLDS[DESIGN_FOLD]["train"]), "only 28-29 March may be used"

    figure_split_by_day(explore)
    figure_normalisation(explore, waveforms)
    figure_padding(explore, waveforms)
    figure_spectrogram_pipeline(explore, waveforms)
    names = ["why_split_by_day", "per_clip_normalisation", "padding_example", "spectrogram_pipeline"]
    print("\nwrote", ", ".join(f"{n}.png" for n in names), "to", FIGURE_DIR)

    if log_wandb:
        import wandb

        from whale.train import PROJECT, wandb_mode
        run = wandb.init(project=PROJECT, name="report_figures", job_type="data", mode=wandb_mode(),
                         config={"days": list(FOLDS[DESIGN_FOLD]["train"])})
        run.log({n: wandb.Image(str(FIGURE_DIR / f"{n}.png")) for n in names})
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wandb", action="store_true", help="also log the figures to W&B")
    main(parser.parse_args().wandb)
