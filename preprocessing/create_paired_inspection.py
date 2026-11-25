import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path("datasets/raw/PCB_Defect")
IMAGE_DIR = ROOT / "images"
ANNOTATION_FILE = ROOT / "annotation/_annotations.coco.json"

OUTPUT_DIR = Path("datasets/processed/paired_inspection")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(ANNOTATION_FILE) as f:
    data = json.load(f)

images = {img["id"]: img for img in data["images"]}

categories = {
    c["id"]: c["name"]
    for c in data["categories"]
    if c["name"] != "detecting-pcb-defects"
}

grouped = defaultdict(list)

for ann in data["annotations"]:
    if ann["category_id"] in categories:
        grouped[ann["category_id"]].append(ann)


def fit_image(image, width, height):
    """Resize image while preserving aspect ratio."""
    h, w = image.shape[:2]

    scale = min(width / w, height / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    )

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    x_offset = (width - new_w) // 2
    y_offset = (height - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    return canvas, scale, x_offset, y_offset


PANEL_W = 750
PANEL_H = 750
HEADER_H = 100

print("Creating original PCB + defect crop pairs...\n")

for category_id, class_name in categories.items():

    annotations = grouped[category_id]

    class_dir = OUTPUT_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    # Select five examples from this class
    selected = annotations[:5]

    for index, ann in enumerate(selected, start=1):

        image_info = images[ann["image_id"]]
        filename = image_info["file_name"]

        image_path = IMAGE_DIR / filename
        original = cv2.imread(str(image_path))

        if original is None:
            print(f"Cannot load {filename}")
            continue

        x, y, w, h = ann["bbox"]

        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        # Context around the defect
        margin_x = max(int(w * 3), 30)
        margin_y = max(int(h * 3), 30)

        cx1 = max(0, x1 - margin_x)
        cy1 = max(0, y1 - margin_y)
        cx2 = min(original.shape[1], x2 + margin_x)
        cy2 = min(original.shape[0], y2 + margin_y)

        crop = original[cy1:cy2, cx1:cx2].copy()

        # Mark defect location on original PCB
        original_marked = original.copy()

        cv2.rectangle(
            original_marked,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            8
        )

        # Mark defect boundary on crop
        cv2.rectangle(
            crop,
            (x1 - cx1, y1 - cy1),
            (x2 - cx1, y2 - cy1),
            (0, 255, 0),
            5
        )

        left, _, _, _ = fit_image(
            original_marked, PANEL_W, PANEL_H
        )

        right, _, _, _ = fit_image(
            crop, PANEL_W, PANEL_H
        )

        # Combined inspection canvas
        result = np.zeros(
            (HEADER_H + PANEL_H, PANEL_W * 2, 3),
            dtype=np.uint8
        )

        result[HEADER_H:, :PANEL_W] = left
        result[HEADER_H:, PANEL_W:] = right

        cv2.putText(
            result,
            f"CLASS: {class_name} | IMAGE: {filename}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            result,
            f"Annotation ID: {ann['id']} | Original PCB",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            result,
            "Enlarged defect crop",
            (PANEL_W + 20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        output_path = class_dir / f"{class_name}_{index}.jpg"

        cv2.imwrite(
            str(output_path),
            result,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        print(f"{class_name:20s} | {filename} | {output_path}")

print("\nPaired inspection images generated.")
