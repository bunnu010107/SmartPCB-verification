import json
import cv2
from pathlib import Path

ROOT = Path("datasets/raw/PCB_Defect")
IMAGE_DIR = ROOT / "images"
ANNOTATION_FILE = ROOT / "annotation/_annotations.coco.json"
OUTPUT_DIR = Path("datasets/processed/annotation_preview")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(ANNOTATION_FILE) as f:
    data = json.load(f)

images = {img["id"]: img for img in data["images"]}

categories = {
    cat["id"]: cat["name"]
    for cat in data["categories"]
    if cat["name"] != "detecting-pcb-defects"
}

annotations_by_class = {}

for ann in data["annotations"]:
    category_id = ann["category_id"]

    if category_id in categories:
        annotations_by_class.setdefault(category_id, []).append(ann)

print("Generating class-specific annotation previews...\n")

for category_id, class_name in categories.items():

    class_annotations = annotations_by_class.get(category_id, [])

    if not class_annotations:
        print(f"WARNING: No annotations for {class_name}")
        continue

    # Group this class's annotations by image
    grouped = {}

    for ann in class_annotations:
        grouped.setdefault(ann["image_id"], []).append(ann)

    # Select the image containing the most annotations of this class
    selected_image_id = max(
        grouped,
        key=lambda image_id: len(grouped[image_id])
    )

    image_info = images[selected_image_id]
    image_path = IMAGE_DIR / image_info["file_name"]

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not load: {image_path}")
        continue

    # Draw ONLY the current defect class
    selected_annotations = grouped[selected_image_id]

    for ann in selected_annotations:

        x, y, w, h = ann["bbox"]

        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        cv2.putText(
            image,
            class_name,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Resize for viewing only
    preview = cv2.resize(
        image,
        (1200, int(image.shape[0] * 1200 / image.shape[1]))
    )

    output_path = OUTPUT_DIR / f"{class_name}_preview.jpg"

    cv2.imwrite(str(output_path), preview)

    print(f"Class: {class_name}")
    print(f"Source: {image_info['file_name']}")
    print(f"Boxes drawn: {len(selected_annotations)}")
    print(f"Saved: {output_path}\n")

print("Class-specific visualization complete.")
