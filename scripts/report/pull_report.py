"""Read-only dump of a W&B report's blocks. Never saves anything back to W&B.

    $env:WANDB_BASE_URL = "https://api.forge.coreweave.com"
    python scripts/report/pull_report.py                 # the current working draft (DRAFT_URL)
    python scripts/report/pull_report.py <report URL>    # any other report; the Forge '/wandb/' segment is dropped automatically
"""
import argparse
import json

import wandb_workspaces.reports.v2 as wr

DRAFT_URL = ("https://forge.coreweave.com/roisegel-personal/whale-upcall/reports/"
             "DeepVoice-Assignment-draft-15-11-13-20--VmlldzoxODMwODgzMA")

ap = argparse.ArgumentParser()
ap.add_argument("url", nargs="?", default=DRAFT_URL)
ap.add_argument("--out", default="report_blocks.txt")
args = ap.parse_args()

url = args.url.replace("forge.coreweave.com/wandb/", "forge.coreweave.com/")  # from_url breaks on the extra segment
report = wr.Report.from_url(url)
print(report.title, len(report.blocks), "blocks")

# 1) human-readable dump of every block (text, headings, tables, panel grids)
with open(args.out, "w", encoding="utf-8") as f:
    for i, b in enumerate(report.blocks):
        f.write(f"--- [{i}] {type(b).__name__} ---\n{b}\n\n")
print("wrote", args.out)

# 2) raw spec, if available, for lossless reference
spec = getattr(report, "spec", None)
if spec is not None:
    with open(args.out.replace(".txt", "_spec.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, default=str)
