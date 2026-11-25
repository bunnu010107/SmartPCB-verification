import json
import cv2
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent

ANNOTATION_FILE = (
    ROOT / "datasets/raw/PCB_Defect/annotation/_annotations.coco.json"
)

IMAGE_DIR = ROOT / "datasets/raw/PCB_Defect/images"


def main():
    with open(ANNOTATION_FILE, "r") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    categories = {
        c["id"]: c["name"]
        for c in data["categories"]
        if c["id"] != 0
    }

    class_counts = Counter()
    image_class_counts = defaultdict(Counter)
    annotations_per_image = Counter()

    invalid_boxes = []
    missing_images = []
    unreadable_images = []
    dimension_mismatches = []
    unknown_categories = []

    image_ids = [img["id"] for img in images]
    filenames = [img["file_name"] for img in images]

    print("=" * 65)
    print("              PCB DATASET QUALITY AUDIT")
    print("=" * 65)

    print(f"\nTotal images:       {len(images)}")
    print(f"Total annotations:  {len(annotations)}")
    print(f"Defect classes:     {len(categories)}")

    # --------------------------------------------------
    # Annotation analysis
    # --------------------------------------------------

    image_lookup = {img["id"]: img for img in images}

    for ann in annotations:
        category_id = ann["category_id"]
        image_id = ann["image_id"]

        if category_id not in categories:
            unknown_categories.append(ann["id"])
            continue

        class_name = categories[category_id]

        class_counts[class_name] += 1
        annotations_per_image[image_id] += 1
        image_class_counts[image_id][class_name] += 1

        image_info = image_lookup.get(image_id)

        if image_info is None:
            invalid_boxes.append((ann["id"], "Unknown image ID"))
            continue

        bbox = ann.get("bbox")

        if not bbox or len(bbox) != 4:
            invalid_boxes.append((ann["id"], "Invalid bbox format"))
            continue

        x, y, width, height = bbox

        image_width = image_info["width"]
        image_height = image_info["height"]

        if (
            width <= 0 or height <= 0
            or x < 0 or y < 0
            or x + width > image_width
            or y + height > image_height
        ):
            invalid_boxes.append((ann["id"], bbox))

    # --------------------------------------------------
    # Image validation
    # --------------------------------------------------

    for img in images:
        image_path = IMAGE_DIR / img["file_name"]

        if not image_path.exists():
            missing_images.append(img["file_name"])
            continue

        actual = cv2.imread(str(image_path))

        if actual is None:
            unreadable_images.append(img["file_name"])
            continue

        actual_height, actual_width = actual.shape[:2]

        if (
            actual_width != img["width"]
            or actual_height != img["height"]
        ):
            dimension_mismatches.append(
                (img["file_name"], img["width"], img["height"],
                 actual_width, actual_height)
            )

    # --------------------------------------------------
    # Dataset report
    # --------------------------------------------------

    print("\n" + "-" * 65)
    print("DEFECT CLASS DISTRIBUTION")
    print("-" * 65)

    for class_name in sorted(categories.values()):
        print(f"{class_name:20s}: {class_counts[class_name]}")

    print("\n" + "-" * 65)
    print("ANNOTATIONS PER IMAGE")
    print("-" * 65)

    annotated_images = sum(
        1 for img in images
        if annotations_per_image[img["id"]] > 0
    )

    empty_images = len(images) - annotated_images

    print(f"Images with annotations: {annotated_images}")
    print(f"Images without annotations: {empty_images}")

    if annotations_per_image:
        values = list(annotations_per_image.values())
        print(f"Minimum annotations/image: {min(values)}")
        print(f"Maximum annotations/image: {max(values)}")
        print(f"Average annotations/image: {sum(values) / len(images):.2f}")

    print("\n" + "-" * 65)
    print("IMAGE AND ANNOTATION VALIDATION")
    print("-" * 65)

    print(f"Duplicate image IDs:       {len(image_ids) - len(set(image_ids))}")
    print(f"Duplicate filenames:       {len(filenames) - len(set(filenames))}")
    print(f"Missing image files:       {len(missing_images)}")
    print(f"Unreadable images:         {len(unreadable_images)}")
    print(f"Dimension mismatches:      {len(dimension_mismatches)}")
    print(f"Invalid bounding boxes:    {len(invalid_boxes)}")
    print(f"Unknown category IDs:      {len(unknown_categories)}")

    print("\n" + "=" * 65)
    print("AUDIT COMPLETED")
    print("=" * 65)

    if invalid_boxes:
        print("\nInvalid bounding boxes (first 10):")
        for item in invalid_boxes[:10]:
            print(item)

    if missing_images:
        print("\nMissing images (first 10):")
        print(missing_images[:10])

    if dimension_mismatches:
        print("\nDimension mismatches (first 10):")
        for item in dimension_mismatches[:10]:
            print(item)


if __name__ == "__main__":
    main()
