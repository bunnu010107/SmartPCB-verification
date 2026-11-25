"""Match predicted rotated boxes to ground truth at several confidence thresholds.

Usage (from the project root):
    python detection/operating_point.py --weights detection/weights/smartpcb_v1_1024_grouped.pt --imgsz 1024 --split val

Choose thresholds on the VAL split. Use --split test only to report a final number.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "datasets" / "processed" / "v2_obb_grouped"
NAMES = ["missing_pad", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
THRESHOLDS = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
IOU_MATCH = 0.5


def poly_iou(p1, p2):
    """IoU of two convex quadrilaterals, each a (4, 2) float array in pixels."""
    h1 = cv2.convexHull(p1.astype(np.float32))   # convex hull gives a consistent point order
    h2 = cv2.convexHull(p2.astype(np.float32))
    a1, a2 = cv2.contourArea(h1), cv2.contourArea(h2)
    if a1 <= 0 or a2 <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(h1, h2)
    return float(inter) / (a1 + a2 - inter + 1e-9)


def load_gt(label_path, w, h):
    """Read a YOLO-OBB label file: class x1 y1 x2 y2 x3 y3 x4 y4 (normalised)."""
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 9:
            continue
        pts = np.array(parts[1:], dtype=np.float32).reshape(4, 2) * np.array([w, h], dtype=np.float32)
        boxes.append((int(float(parts[0])), pts))
    return boxes


def match(gt, preds, thr):
    """Greedy matching for one image; returns per-class TP, FP, FN arrays."""
    tp, fp, fn = np.zeros(6, int), np.zeros(6, int), np.zeros(6, int)
    used = set()
    for cls, conf, poly in sorted((p for p in preds if p[1] >= thr), key=lambda p: -p[1]):
        best_iou, best_j = 0.0, None
        for j, (gcls, gpoly) in enumerate(gt):
            if j in used or gcls != cls:
                continue
            iou = poly_iou(poly, gpoly)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j is not None and best_iou >= IOU_MATCH:
            used.add(best_j)
            tp[cls] += 1
        else:
            fp[cls] += 1
    for j, (gcls, _) in enumerate(gt):
        if j not in used:
            fn[gcls] += 1
    return tp, fp, fn


def parse_args():
    p = argparse.ArgumentParser(description="Operating-point analysis")
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--data-dir", default=str(SPLIT_DIR), help="folder with images/ and labels/ (default: grouped split)")
    return p.parse_args()


def main():
    a = parse_args()
    torch.set_num_threads(4)
    if a.split == "test":
        print("NOTE: test split - report only, do not tune thresholds on it.")

    model = YOLO(str(Path(a.weights).resolve()))
    label_dir = Path(a.data_dir) / "labels" / a.split
    images = sorted((Path(a.data_dir) / "images" / a.split).glob("*.jpg"))

    # Predict once at a low confidence; thresholds are applied afterwards.
    data = []
    for img in images:
        r = model.predict(source=str(img), imgsz=a.imgsz, conf=0.05, device="cpu", verbose=False)[0]
        h, w = r.orig_shape
        gt = load_gt(label_dir / f"{img.stem}.txt", w, h)
        preds = []
        if r.obb is not None and len(r.obb) > 0:
            polys = r.obb.xyxyxyxy.cpu().numpy()
            confs = r.obb.conf.cpu().numpy()
            clss = r.obb.cls.cpu().numpy().astype(int)
            preds = list(zip(clss, confs, polys))
        data.append((gt, preds))
    n_img = len(data)
    n_gt = sum(len(gt) for gt, _ in data)
    print(f"weights: {a.weights} | imgsz: {a.imgsz} | split: {a.split} | images: {n_img} | GT boxes: {n_gt}\n")

    print("thr  |  TP   FP   FN |  prec   rec    F1  | FP/img  missed/img")
    rows = []
    for thr in THRESHOLDS:
        tp, fp, fn = np.zeros(6, int), np.zeros(6, int), np.zeros(6, int)
        for gt, preds in data:
            t, f, n = match(gt, preds, thr)
            tp += t; fp += f; fn += n
        TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
        prec = TP / max(TP + FP, 1)
        rec = TP / max(TP + FN, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append((thr, tp, fn, f1))
        print(f"{thr:.2f} | {TP:4d} {FP:4d} {FN:4d} | {prec:.3f} {rec:.3f} {f1:.3f} | {FP / n_img:5.2f}   {FN / n_img:5.2f}")

    print("\nper-class recall at each threshold")
    print("thr  | " + " ".join(f"{n[:10]:>10}" for n in NAMES))
    for thr, tp, fn, _ in rows:
        rec = [tp[c] / max(tp[c] + fn[c], 1) for c in range(6)]
        print(f"{thr:.2f} | " + " ".join(f"{x:10.3f}" for x in rec))

    best = max(rows, key=lambda r: r[3])
    print(f"\nthreshold with the best overall F1: {best[0]:.2f} (F1 {best[3]:.3f})")


if __name__ == "__main__":
    main()
