"""Build a contact sheet of images that share a filename prefix.

Usage (from the project root):
    python preprocessing/make_prefix_montage.py 52
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "datasets" / "processed" / "v2_obb_split" / "images"
OUT_DIR = ROOT / "runs" / "diagnostics"
TILE_W, TILE_H, HEADER = 480, 360, 30
COLS = 4
MAX_IMAGES = 8


def make_tile(path, split):
    """Resize one image into a fixed-size tile with a text header."""
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Cannot read {path}")
    h, w = img.shape[:2]
    scale = min(TILE_W / w, (TILE_H - HEADER) / h)
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    tile = np.full((TILE_H, TILE_W, 3), 255, np.uint8)
    tile[HEADER:HEADER + small.shape[0], :small.shape[1]] = small
    label = f"{path.name} [{split}] {w}x{h}"
    cv2.putText(tile, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return tile


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python preprocessing/make_prefix_montage.py <first filename part, e.g. 52>")
    prefix = sys.argv[1]

    # Group the matching images by their first TWO filename parts (e.g. "52-1", "52-2").
    groups = {}
    for split in ("train", "val", "test"):
        for p in sorted((IMAGES / split).glob(f"{prefix}-*.jpg")):
            key = "-".join(p.stem.split("-")[:2])
            groups.setdefault(key, []).append((p, split))
    if not groups:
        sys.exit(f"No images found starting with '{prefix}-'")

    # Take up to 2 images per group so different groups are all visible.
    chosen = []
    for key in sorted(groups):
        chosen.extend(groups[key][:2])
    chosen = chosen[:MAX_IMAGES]

    tiles = [make_tile(p, s) for p, s in chosen]
    while len(tiles) % COLS:
        tiles.append(np.full((TILE_H, TILE_W, 3), 255, np.uint8))
    rows = [np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]
    sheet = np.vstack(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"prefix_{prefix}.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])

    print("images with this prefix:", sum(len(v) for v in groups.values()))
    for key in sorted(groups):
        print(" group", key, ":", [p.name for p, _ in groups[key]])
    print("saved:", out)


if __name__ == "__main__":
    main()
