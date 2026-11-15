"""Deterministic quality-control rules for SmartPCB.

No ML libraries are imported, so this logic is easy to port to Kotlin later.
A detection is a dict: {"class_id": int, "confidence": float, "polygon": [[x, y], x4]}
with the polygon in image pixels.

Rules (version 1):
  HOLD   - at least one detection with confidence >= hold_conf
  REVIEW - no HOLD, but at least one detection with review_conf <= confidence < hold_conf
  PASS   - nothing at or above review_conf
No severity is assigned yet: acceptance criteria have not been agreed.
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def validate_config(cfg):
    if not (0.0 <= cfg["review_conf"] <= cfg["hold_conf"] <= 1.0):
        raise ValueError("need 0 <= review_conf <= hold_conf <= 1")
    if not cfg["class_names"]:
        raise ValueError("class_names must not be empty")
    return cfg


def load_config(path):
    return validate_config(json.loads(Path(path).read_text()))


def _describe(det, class_names, tier):
    """Turn one raw detection into a JSON-friendly defect record."""
    poly = det["polygon"]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    side_a = math.dist(poly[0], poly[1])
    side_b = math.dist(poly[1], poly[2])
    return {
        "class": class_names[det["class_id"]],
        "confidence": round(float(det["confidence"]), 4),
        "tier": tier,
        "center_px": [round(sum(xs) / 4, 1), round(sum(ys) / 4, 1)],
        "size_px": [round(max(side_a, side_b), 1), round(min(side_a, side_b), 1)],
        "polygon_px": [[round(float(x), 1), round(float(y), 1)] for x, y in poly],
    }


def build_report(detections, cfg, inspection_id, image_name, image_size, timestamp=None):
    """Apply the rules to one board's detections and return the JSON-ready report."""
    hold, review = cfg["hold_conf"], cfg["review_conf"]

    defects = []
    for det in sorted(detections, key=lambda d: -float(d["confidence"])):
        conf = float(det["confidence"])
        if conf >= hold:
            defects.append(_describe(det, cfg["class_names"], "confirmed"))
        elif conf >= review:
            defects.append(_describe(det, cfg["class_names"], "possible"))

    n_confirmed = sum(d["tier"] == "confirmed" for d in defects)
    n_possible = len(defects) - n_confirmed
    if n_confirmed:
        status = "HOLD"
        reason = f"{n_confirmed} confirmed defect(s) at confidence >= {hold}"
    elif n_possible:
        status = "REVIEW"
        reason = f"{n_possible} possible defect(s) between {review} and {hold}; operator check needed"
    else:
        status = "PASS"
        reason = f"no detection at confidence >= {review}"

    counts = {name: 0 for name in cfg["class_names"]}
    for d in defects:
        counts[d["class"]] += 1

    return {
        "inspection_id": inspection_id,
        "timestamp_utc": timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image": image_name,
        "image_size_px": [int(image_size[0]), int(image_size[1])],
        "model": {
            "name": cfg["model_name"],
            "imgsz": cfg["imgsz"],
            "weights_sha256": cfg["weights_sha256"],
        },
        "thresholds": {"hold_conf": hold, "review_conf": review},
        "board_status": status,
        "reason": reason,
        "defect_counts": counts,
        "defects": defects,
    }
