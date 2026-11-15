"""Run the detector on one image and return detections in the format the QC rules expect."""
import hashlib

from ultralytics import YOLO


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(weights):
    return YOLO(str(weights))


def detect(model, image_path, imgsz, conf):
    """Return (detections, (width, height)). Each detection:
    {"class_id": int, "confidence": float, "polygon": [[x, y] x4]} in original image pixels."""
    r = model.predict(source=str(image_path), imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]
    height, width = r.orig_shape
    detections = []
    if r.obb is not None and len(r.obb) > 0:
        polys = r.obb.xyxyxyxy.cpu().numpy()
        confs = r.obb.conf.cpu().numpy()
        classes = r.obb.cls.cpu().numpy().astype(int)
        for c, s, poly in zip(classes, confs, polys):
            detections.append({"class_id": int(c), "confidence": float(s), "polygon": poly.tolist()})
    return detections, (width, height)
