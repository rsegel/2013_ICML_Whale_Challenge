"""Parse clip filenames into a table.

Filenames look like::

    20090328_000000_002s3ms_TRAIN0_1.aif     (train: trailing _1 / _0 is the label)
    20090404_000000_012s0ms_Test0.aif        (test: no label)

Fields: recording date, start time of the 15-minute chunk (HHMMSS), offset of
the clip inside that chunk, a running clip index, and the label.
"""

import re
from pathlib import Path

import pandas as pd

_PATTERN = re.compile(
    r"^(?P<date>\d{8})_(?P<chunk>\d{6})_(?P<sec>\d+)s(?P<frac>\d+)ms_"
    r"(?:TRAIN|Test)(?P<index>\d+)(?:_(?P<label>[01]))?\.aif$"
)


def parse_filename(name: str) -> dict:
    """Return the fields encoded in one clip filename.

    The ``ms`` part of the offset is a single digit in every file, i.e. it is
    tenths of a second, not milliseconds (checked in the data audit).
    """
    match = _PATTERN.match(name)
    if match is None:
        raise ValueError(f"Unrecognised clip filename: {name}")
    label = match["label"]
    return {
        "filename": name,
        "date": match["date"],
        "chunk": match["chunk"],
        "chunk_id": f"{match['date']}_{match['chunk']}",
        "offset_s": int(match["sec"]) + int(match["frac"]) / 10.0,
        "offset_frac_raw": int(match["frac"]),
        "clip_index": int(match["index"]),
        "label": int(label) if label is not None else -1,
    }


def build_manifest(audio_dir: Path) -> pd.DataFrame:
    """One row per .aif file in ``audio_dir``, in running clip-index order."""
    rows = [parse_filename(p.name) for p in audio_dir.glob("*.aif")]
    if not rows:
        raise FileNotFoundError(f"No .aif files in {audio_dir}")
    frame = pd.DataFrame(rows).sort_values("clip_index").reset_index(drop=True)
    if frame["clip_index"].duplicated().any():
        raise ValueError("Duplicate clip indices in filenames")
    return frame
