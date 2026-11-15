"""Export a trained YOLO26n-OBB model to ONNX and print what the exported file expects and returns.

Usage (from the project root):
    python detection/export_onnx.py --weights detection/weights/smartpcb_v1_1024_grouped.pt --imgsz 1024
"""
import argparse
import hashlib
from pathlib import Path

import onnxruntime as ort
from ultralytics import YOLO


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description="Export to ONNX")
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, required=True)
    return p.parse_args()


def main():
    a = parse_args()
    weights = Path(a.weights).resolve()
    # Fixed input size, float32, opset 17 (widely supported by mobile runtimes).
    exported = YOLO(str(weights)).export(
        format="onnx", imgsz=a.imgsz, opset=17, simplify=True, dynamic=False, half=False, device="cpu"
    )
    onnx_path = Path(exported)
    print("\nexported:", onnx_path)
    print("size MB:", round(onnx_path.stat().st_size / 1e6, 2))
    print("sha256:", sha256(onnx_path))

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for i in sess.get_inputs():
        print("input :", i.name, i.shape, i.type)
    for o in sess.get_outputs():
        print("output:", o.name, o.shape, o.type)


if __name__ == "__main__":
    main()
