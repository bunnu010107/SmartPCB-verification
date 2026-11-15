"""Print a short training summary for a run under runs/train.

Usage (from the project root):
    python detection/summarize_run.py grouped_640
"""
import csv
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get(row, suffix):
    for k, v in row.items():
        if k.endswith(suffix):
            return float(v)


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python detection/summarize_run.py <run name>")
    name = sys.argv[1]
    pattern = str(ROOT / "runs" / "train" / f"{name}*" / "results.csv")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not matches:
        sys.exit(f"No results.csv found for '{name}'")
    path = matches[-1]  # newest folder, so a "-2" suffix is handled
    rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(open(path))]
    best = max(range(len(rows)), key=lambda i: get(rows[i], "metrics/mAP50-95(B)")) + 1
    print("file:", path)
    print("epochs logged:", len(rows), "| best mAP50-95 epoch:", best)
    print("epoch | train cls | val cls | recall | mAP50 | mAP50-95")
    for i in range(5, len(rows) + 1, 5):
        r = rows[i - 1]
        print(i, "|", round(get(r, "train/cls_loss"), 3), "|", round(get(r, "val/cls_loss"), 3), "|",
              round(get(r, "metrics/recall(B)"), 3), "|", round(get(r, "metrics/mAP50(B)"), 3), "|",
              round(get(r, "metrics/mAP50-95(B)"), 3))


if __name__ == "__main__":
    main()
