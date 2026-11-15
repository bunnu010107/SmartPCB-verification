"""Inspect PCB images: detect -> QC rules -> JSON report + annotated image + SQLite record.

Usage (from the project root):
    python inspect_board.py IMAGE [IMAGE ...]
Options: --weights, --config, --out
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from detection.predict import detect, load_model, sha256_file  # noqa: E402
from quality_control.database import next_inspection_id, open_db, save_report  # noqa: E402
from quality_control.rules import build_report, load_config  # noqa: E402

TIER_COLOR = {"confirmed": (0, 0, 255), "possible": (0, 165, 255)}               # BGR: red, orange
STATUS_COLOR = {"HOLD": (0, 0, 255), "REVIEW": (0, 165, 255), "PASS": (0, 160, 0)}


def annotate(image_path, report, out_path):
    """Draw each reported defect and a status banner on a copy of the image."""
    img = cv2.imread(str(image_path))
    thick = max(2, img.shape[1] // 600)
    scale = max(0.6, img.shape[1] / 2500)
    for d in report["defects"]:
        pts = np.array(d["polygon_px"], dtype=np.int32).reshape(-1, 1, 2)
        color = TIER_COLOR[d["tier"]]
        cv2.polylines(img, [pts], True, color, thick)
        x, y = int(pts[:, 0, 0].min()), int(pts[:, 0, 1].min())
        cv2.putText(img, f'{d["class"]} {d["confidence"]:.2f}', (x, max(y - 6, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
    banner = f'{report["inspection_id"]}  {report["board_status"]}'
    cv2.putText(img, banner, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, scale * 1.6,
                STATUS_COLOR[report["board_status"]], thick + 1)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def parse_args():
    p = argparse.ArgumentParser(description="SmartPCB board inspection")
    p.add_argument("images", nargs="+")
    p.add_argument("--weights", default=str(ROOT / "detection" / "weights" / "smartpcb_v1_1024_grouped.pt"))
    p.add_argument("--config", default=str(ROOT / "quality_control" / "config_v1_1024.json"))
    p.add_argument("--out", default=str(ROOT / "runs" / "inspections"))
    return p.parse_args()


def main():
    a = parse_args()
    cfg = load_config(a.config)
    if sha256_file(a.weights) != cfg["weights_sha256"]:
        sys.exit("Weights checksum does not match the config - wrong model file?")

    torch.set_num_threads(4)
    model = load_model(a.weights)
    out = Path(a.out)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "annotated").mkdir(parents=True, exist_ok=True)
    conn = open_db(out / "inspections.db")

    for image in a.images:
        t0 = time.perf_counter()
        detections, size = detect(model, image, cfg["imgsz"], cfg["review_conf"])
        detect_ms = (time.perf_counter() - t0) * 1000

        iid = next_inspection_id(conn)
        report = build_report(detections, cfg, iid, Path(image).name, size)
        report["timing_ms"] = {"detect": round(detect_ms, 1)}

        (out / "reports" / f"{iid}.json").write_text(json.dumps(report, indent=2))
        annotate(image, report, out / "annotated" / f"{iid}.jpg")
        save_report(conn, report, image)

        n_conf = sum(d["tier"] == "confirmed" for d in report["defects"])
        n_poss = len(report["defects"]) - n_conf
        print(f'{iid} | {Path(image).name} | {report["board_status"]:6} | '
              f'{n_conf} confirmed, {n_poss} possible | detect {detect_ms:.0f} ms')
    conn.close()
    print("outputs in:", out)


if __name__ == "__main__":
    main()
