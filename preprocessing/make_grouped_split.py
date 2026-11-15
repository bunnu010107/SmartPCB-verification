"""Create a group-aware train/val/test split of the PCB V2 OBB dataset.

Images whose filenames share the same first part (e.g. "52-...") always land in
the same split, so no group is shared between train, val and test.

Source (read only): datasets/processed/v2_obb_split
Output (new):       datasets/processed/v2_obb_grouped
Run from anywhere:  python preprocessing/make_grouped_split.py
"""
import csv
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "datasets" / "processed" / "v2_obb_split"
DST = ROOT / "datasets" / "processed" / "v2_obb_grouped"
MANIFEST = ROOT / "datasets" / "splits" / "v2_obb_grouped_manifest.csv"
NAMES = ["missing_pad", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42
TRIALS = 20000


def load_items():
    """Read every image + label of the existing split and count boxes per class."""
    items = []
    for split in RATIOS:
        for img in sorted((SRC / "images" / split).glob("*.jpg")):
            label = SRC / "labels" / split / (img.stem + ".txt")
            counts = Counter()
            for line in label.read_text().splitlines():
                parts = line.split()
                if parts:
                    counts[int(float(parts[0]))] += 1
            items.append({"img": img, "label": label,
                          "group": img.stem.split("-")[0], "counts": counts})
    return items


def assign(groups, rng, total_images):
    """Shuffle the groups, then give each to the split furthest below its image target."""
    order = list(groups)
    rng.shuffle(order)
    current = {s: 0 for s in RATIOS}
    assignment = {}
    for g in order:
        split = max(RATIOS, key=lambda s: 1 - current[s] / (RATIOS[s] * total_images))
        assignment[g] = split
        current[split] += groups[g]["n"]
    return assignment


def score(assignment, groups, total_images, class_totals):
    """Lower is better: distance of image and per-class box counts from the 70/15/15 targets."""
    imgs = Counter()
    boxes = {s: Counter() for s in RATIOS}
    for g, s in assignment.items():
        imgs[s] += groups[g]["n"]
        boxes[s].update(groups[g]["counts"])
    total = 0.0
    for s, r in RATIOS.items():
        total += abs(imgs[s] - r * total_images) / total_images
        for c, t in class_totals.items():
            total += abs(boxes[s][c] - r * t) / t
            if s != "train" and boxes[s][c] == 0:
                total += 10  # every class must appear in val and test
    return total


def main():
    if DST.exists():
        sys.exit(f"{DST} already exists - refusing to overwrite it.")

    items = load_items()
    total_images = len(items)
    class_totals = Counter()
    groups = {}
    for it in items:
        g = groups.setdefault(it["group"], {"n": 0, "counts": Counter()})
        g["n"] += 1
        g["counts"].update(it["counts"])
        class_totals.update(it["counts"])
    print("images:", total_images, "| groups:", len(groups), "| boxes:", sum(class_totals.values()))

    # Try many random group assignments and keep the best-balanced one (fixed seed).
    rng = random.Random(SEED)
    best_score, best = None, None
    for _ in range(TRIALS):
        a = assign(groups, rng, total_images)
        sc = score(a, groups, total_images, class_totals)
        if best_score is None or sc < best_score:
            best_score, best = sc, a
    print("best balance score:", round(best_score, 3))

    # Copy files into the new folder (originals are never modified).
    manifest_rows = []
    for it in items:
        split = best[it["group"]]
        for kind, src in (("images", it["img"]), ("labels", it["label"])):
            out_dir = DST / kind / split
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out_dir / src.name)
        manifest_rows.append([it["img"].stem, it["group"], split, sum(it["counts"].values())])

    yaml_lines = [f"path: {DST}", "", "train: images/train", "val: images/val",
                  "test: images/test", "", "names:"]
    yaml_lines += [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (DST / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "group", "split", "n_boxes"])
        w.writerows(manifest_rows)

    # Report
    members = {s: set() for s in RATIOS}
    for g, s in best.items():
        members[s].add(g)
    print("\nclass order:", NAMES)
    print("split | images | groups | boxes per class")
    for s in RATIOS:
        n_img = sum(groups[g]["n"] for g in members[s])
        boxes = Counter()
        for g in members[s]:
            boxes.update(groups[g]["counts"])
        print(s, "|", n_img, "|", len(members[s]), "|", [boxes[c] for c in range(len(NAMES))])
    assert not (members["train"] & members["val"] or members["train"] & members["test"]
                or members["val"] & members["test"])
    print("groups shared between splits: 0")
    print("yaml:", DST / "data.yaml")
    print("manifest:", MANIFEST)


if __name__ == "__main__":
    main()
