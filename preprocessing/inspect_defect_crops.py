import json
import cv2
from pathlib import Path
from collections import defaultdict

ROOT = Path("datasets/raw/PCB_Defect")
IMAGE_DIR = ROOT / "images"
ANNOTATION_FILE = ROOT / "annotation/_annotations.coco.json"
OUTPUT_DIR = Path("datasets/processed/defect_crops")

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

print("Generating high-resolution defect crops...\n")

for category_id, class_name in categories.items():

    class_dir = OUTPUT_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    annotations = grouped[category_id]

    # Select up to five annotations for inspection
    selected = annotations[:5]

    print(f"{class_name}: {len(selected)} crops")

    for index, ann in enumerate(selected, start=1):

        image_info = images[ann["image_id"]]
        image_path = IMAGE_DIR / image_info["file_name"]

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        x, y, w, h = ann["bbox"]

        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)

        # Add 3x context around the bounding box
        margin_x = max(int(w), 30)
        margin_y = max(int(h), 30)

        crop_x1 = max(0, x1 - margin_x)
        crop_y1 = max(0, y1 - margin_y)
        crop_x2 = min(image.shape[1], x2 + margin_x)
        crop_y2 = min(image.shape[0], y2 + margin_y)

        crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()

        # Draw the original annotation boundary
        cv2.rectangle(
            crop,
            (x1 - crop_x1, y1 - crop_y1),
            (x2 - crop_x1, y2 - crop_y1),
            (0, 255, 0),
            2
        )

        output_path = class_dir / f"{class_name}_{index}.jpg"

        cv2.imwrite(str(output_path), crop)

        print(f"  Saved: {output_path}")

print("\nCrop generation complete.")
