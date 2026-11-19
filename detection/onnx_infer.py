"""Standalone ONNX inference for the SmartPCB OBB model (no Ultralytics in the inference path).

This is the reference for what the Android app must do:
  letterbox -> run model -> decode raw output -> confidence filter -> rotated NMS -> map back to the image.

Usage (from the project root):
    python detection/onnx_infer.py --model detection/weights/smartpcb_v1_1024_grouped.onnx \
        --weights detection/weights/smartpcb_v1_1024_grouped.pt --imgsz 1024 --conf 0.40
"""
import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detection.operating_point import poly_iou  # noqa: E402
from detection.predict import detect, load_model  # noqa: E402

VAL_DIR = ROOT / "datasets" / "processed" / "v2_obb_grouped" / "images" / "val"
NUM_CLASSES = 6
NMS_IOU = 0.7  # Ultralytics' default for predict


def letterbox(img, size):
    """Resize keeping the aspect ratio and pad with grey (114) to size x size, like Ultralytics."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def to_input(canvas):
    """BGR uint8 HxWx3 -> float32 1x3xHxW, RGB, values 0..1."""
    x = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(x[None])


def xywhr_to_polygon(cx, cy, w, h, r):
    """Four corner points of a rotated box (the same formula Ultralytics uses)."""
    c, s = np.cos(r), np.sin(r)
    v1 = np.array([w / 2 * c, w / 2 * s])
    v2 = np.array([-h / 2 * s, h / 2 * c])
    ctr = np.array([cx, cy])
    return np.stack([ctr + v1 + v2, ctr + v1 - v2, ctr - v1 - v2, ctr - v1 + v2])


def rotated_nms(boxes, scores, iou_thr):
    """Greedy NMS for rotated boxes (cx, cy, w, h, angle); returns the indices to keep."""
    polys = [xywhr_to_polygon(*b) for b in boxes]
    radius = [0.5 * math.hypot(b[2], b[3]) for b in boxes]
    keep = []
    for i in np.argsort(-np.asarray(scores)):
        i = int(i)
        suppressed = False
        for j in keep:
            if math.hypot(boxes[i][0] - boxes[j][0], boxes[i][1] - boxes[j][1]) > radius[i] + radius[j]:
                continue  # too far apart to overlap, skip the expensive test
            if poly_iou(polys[i], polys[j]) >= iou_thr:
                suppressed = True
                break
        if not suppressed:
            keep.append(i)
    return keep


def decode(raw, conf, nms_iou):
    """raw: model output (11, N) = cx, cy, w, h, 6 class scores, angle (letterboxed pixels).
    Returns [(class_id, score, cx, cy, w, h, angle), ...] after the confidence filter and per-class NMS."""
    pred = raw.T
    scores = pred[:, 4:4 + NUM_CLASSES]
    cls = scores.argmax(axis=1)
    best = scores.max(axis=1)
    keep = best >= conf
    pred, cls, best = pred[keep], cls[keep], best[keep]
    results = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        boxes = [(pred[i, 0], pred[i, 1], pred[i, 2], pred[i, 3], pred[i, -1]) for i in idx]
        for k in rotated_nms(boxes, best[idx], nms_iou):
            i = idx[k]
            results.append((int(c), float(best[i]), float(pred[i, 0]), float(pred[i, 1]),
                            float(pred[i, 2]), float(pred[i, 3]), float(pred[i, -1])))
    return results


def infer(session, input_name, img, imgsz, conf):
    """Full pipeline on one BGR image. Returns detections (original-image pixels), timings, raw output."""
    t0 = time.perf_counter()
    canvas, r, left, top = letterbox(img, imgsz)
    x = to_input(canvas)
    t1 = time.perf_counter()
    raw = session.run(None, {input_name: x})[0][0]
    t2 = time.perf_counter()
    found = decode(raw, conf, NMS_IOU)
    dets = []
    for cls, score, cx, cy, w, h, ang in found:
        cx, cy = (cx - left) / r, (cy - top) / r   # undo padding and scaling
        w, h = w / r, h / r
        dets.append({"class_id": cls, "confidence": score,
                     "polygon": xywhr_to_polygon(cx, cy, w, h, ang).tolist()})
    t3 = time.perf_counter()
    return dets, {"pre": (t1 - t0) * 1000, "run": (t2 - t1) * 1000, "post": (t3 - t2) * 1000}, raw


def compare(ref, other, iou_thr=0.5):
    """Greedy one-to-one match of same-class detections by polygon IoU. Returns (matched, conf differences)."""
    used, diffs = set(), []
    for a in sorted(ref, key=lambda d: -d["confidence"]):
        best, best_j = 0.0, None
        for j, b in enumerate(other):
            if j in used or b["class_id"] != a["class_id"]:
                continue
            iou = poly_iou(np.array(a["polygon"]), np.array(b["polygon"]))
            if iou > best:
                best, best_j = iou, j
        if best_j is not None and best >= iou_thr:
            used.add(best_j)
            diffs.append(abs(a["confidence"] - other[best_j]["confidence"]))
    return len(diffs), diffs


def parse_args():
    p = argparse.ArgumentParser(description="Standalone ONNX inference check")
    p.add_argument("--model", required=True, help=".onnx file")
    p.add_argument("--weights", required=True, help=".pt file to compare against")
    p.add_argument("--imgsz", type=int, required=True)
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit", type=int, default=35)
    return p.parse_args()


def main():
    a = parse_args()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = a.threads
    sess = ort.InferenceSession(a.model, sess_options=opts, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    sess.run(None, {name: np.zeros((1, 3, a.imgsz, a.imgsz), np.float32)})  # warm-up
    torch_model = load_model(a.weights)

    images = sorted(VAL_DIR.glob("*.jpg"))[: a.limit]
    n_ref = n_onnx = n_match = 0
    all_diffs, times = [], []
    for n, path in enumerate(images):
        img = cv2.imread(str(path))
        dets, tm, raw = infer(sess, name, img, a.imgsz, a.conf)
        if n == 0:
            print("raw output shape:", raw.shape)
            labels = ["cx", "cy", "w", "h"] + [f"cls{i}" for i in range(NUM_CLASSES)] + ["angle"]
            for k, lab in enumerate(labels):
                print(f"  {lab:>6}: min {raw[k].min():9.3f}  max {raw[k].max():9.3f}")
        ref, _ = detect(torch_model, path, a.imgsz, a.conf)
        matched, diffs = compare(ref, dets)
        n_ref += len(ref); n_onnx += len(dets); n_match += matched
        all_diffs += diffs
        times.append(tm)
        print(f"{path.name:12} | torch {len(ref):3d} | onnx {len(dets):3d} | matched {matched:3d}")

    print(f"\nimages: {len(images)} | conf >= {a.conf}")
    print(f"PyTorch detections: {n_ref} | ONNX detections: {n_onnx} | matched: {n_match}")
    print(f"share of PyTorch detections found by ONNX: {n_match / max(n_ref, 1):.3f}")
    print(f"share of ONNX detections confirmed by PyTorch: {n_match / max(n_onnx, 1):.3f}")
    if all_diffs:
        print(f"mean |confidence difference| on matched pairs: {np.mean(all_diffs):.3f}")
    m = {k: float(np.mean([t[k] for t in times])) for k in ("pre", "run", "post")}
    print(f"mean ms per image: preprocess {m['pre']:.1f} | model run {m['run']:.1f} | "
          f"decode+NMS {m['post']:.1f} | total {sum(m.values()):.1f}")


if __name__ == "__main__":
    main()
