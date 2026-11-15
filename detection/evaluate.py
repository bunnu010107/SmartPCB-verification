"""Evaluate a trained YOLO26n-OBB model on the val or test split of the grouped dataset.

Usage (from the project root):
    python detection/evaluate.py --weights detection/weights/smartpcb_v1_1024_grouped.pt \
        --imgsz 1024 --split val --name check_val_1024
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "datasets" / "processed" / "v2_obb_grouped" / "data.yaml"
EVAL_DIR = ROOT / "runs" / "eval"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained model")
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, required=True, help="use the size the model was trained at")
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--name", required=True, help="output folder name under runs/eval")
    return p.parse_args()


def main():
    a = parse_args()
    torch.set_num_threads(4)
    weights = Path(a.weights).resolve()

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(DATA_YAML), split=a.split, imgsz=a.imgsz,
        batch=2, workers=0, device="cpu",
        plots=True, project=str(EVAL_DIR), name=a.name, exist_ok=False,
    )

    result = {
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "data": str(DATA_YAML),
        "split": a.split,
        "imgsz": a.imgsz,
        "overall": {k: float(v) for k, v in metrics.results_dict.items()},
        "per_class": {},
    }
    try:
        for i, class_idx in enumerate(metrics.box.ap_class_index):
            p, r, ap50, ap = metrics.box.class_result(i)
            result["per_class"][metrics.names[int(class_idx)]] = {
                "precision": float(p), "recall": float(r),
                "mAP50": float(ap50), "mAP50-95": float(ap),
            }
    except Exception as e:  # the console table printed above is still the source of truth
        print("note: per-class JSON skipped:", e)

    out = Path(metrics.save_dir) / "metrics.json"
    out.write_text(json.dumps(result, indent=2))
    print("metrics saved to:", out)


if __name__ == "__main__":
    main()
