import argparse
import json
import cv2
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANNOTATION_FILE = (
    PROJECT_ROOT / "datasets/raw/PCB_Defect/annotation/_annotations.coco.json"
)

OUTPUT_ROOT = PROJECT_ROOT / "datasets/processed/single_inspection"


def load_annotations():
    with open(ANNOTATION_FILE, "r") as file:
        data = json.load(file)

    categories = {
        category["id"]: category["name"]
        for category in data["categories"]
        if category["name"] != "detecting-pcb-defects"
    }

    return data, categories


def find_image_annotations(data, image_path):
    image_name = image_path.name

    image_info = next(
        (
            image for image in data["images"]
            if Path(image["file_name"]).name == image_name
        ),
        None
    )

    if image_info is None:
        raise ValueError(
            f"Image '{image_name}' was not found in the COCO annotations."
        )

    annotations = [
        annotation
        for annotation in data["annotations"]
        if annotation["image_id"] == image_info["id"]
    ]

    return image_info, annotations


def create_inspection(image_path, padding_ratio=0.75, scale=2.0):

    data, categories = load_annotations()

    image_path = image_path.resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    original = cv2.imread(str(image_path))

    if original is None:
        raise ValueError("OpenCV could not read this image.")

    height, width = original.shape[:2]

    image_info, annotations = find_image_annotations(data, image_path)

    output_dir = OUTPUT_ROOT / image_path.stem
    crop_dir = output_dir / "defect_crops"

    crop_dir.mkdir(parents=True, exist_ok=True)

    annotated = original.copy()

    report = {
        "image": image_path.name,
        "image_dimensions": {
            "width": width,
            "height": height
        },
        "inspection_source": "COCO ground-truth annotations",
        "inspection_type": "Dataset annotation verification",
        "total_defects": len(annotations),
        "defects": []
    }

    print("\n" + "=" * 60)
    print("           SINGLE PCB IMAGE INSPECTION")
    print("=" * 60)

    print(f"Image: {image_path.name}")
    print(f"Dimensions: {width} x {height}")
    print(f"Ground-truth annotations: {len(annotations)}")

    for index, annotation in enumerate(annotations, start=1):

        category_id = annotation["category_id"]

        if category_id not in categories:
            print(f"Skipping unknown category ID: {category_id}")
            continue

        class_name = categories[category_id]

        x, y, box_width, box_height = annotation["bbox"]

        x1 = max(0, int(round(x)))
        y1 = max(0, int(round(y)))
        x2 = min(width, int(round(x + box_width)))
        y2 = min(height, int(round(y + box_height)))

        if x2 <= x1 or y2 <= y1:
            print(f"Skipping invalid bounding box: {annotation['id']}")
            continue

        # Draw the bounding box on the original image.
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        label = f"{class_name} | ID {annotation['id']}"

        cv2.putText(
            annotated,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        # Extract an enlarged crop with surrounding PCB context.
        padding_x = int((x2 - x1) * padding_ratio)
        padding_y = int((y2 - y1) * padding_ratio)

        crop_x1 = max(0, x1 - padding_x)
        crop_y1 = max(0, y1 - padding_y)
        crop_x2 = min(width, x2 + padding_x)
        crop_y2 = min(height, y2 + padding_y)

        crop = original[crop_y1:crop_y2, crop_x1:crop_x2]

        enlarged_crop = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        crop_filename = (
            f"{index:03d}_{class_name}_annotation_{annotation['id']}.jpg"
        )

        crop_path = crop_dir / crop_filename

        if not cv2.imwrite(str(crop_path), enlarged_crop):
            raise IOError(f"Could not save crop: {crop_path}")

        report["defects"].append({
            "defect_number": index,
            "annotation_id": annotation["id"],
            "class": class_name,
            "category_id": category_id,
            "bounding_box_xywh": [
                round(float(x), 2),
                round(float(y), 2),
                round(float(box_width), 2),
                round(float(box_height), 2)
            ],
            "crop_file": f"defect_crops/{crop_filename}",
            "confidence": None,
            "label_source": "ground_truth"
        })

        print(
            f"{index:03d}. {class_name:<18} "
            f"Annotation ID: {annotation['id']}"
        )

    # Save the original PCB with all annotations.
    annotated_path = output_dir / "annotated_original.jpg"

    if not cv2.imwrite(str(annotated_path), annotated):
        raise IOError("Could not save annotated PCB image.")

    # Save structured inspection report.
    report_path = output_dir / "inspection_report.json"

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    print("\n" + "-" * 60)
    print("INSPECTION COMPLETED")
    print("-" * 60)
    print(f"Defects processed: {len(report['defects'])}")
    print(f"Annotated PCB:     {annotated_path}")
    print(f"Defect crops:      {crop_dir}")
    print(f"JSON report:       {report_path}")
    print("=" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Inspect one PCB image using COCO ground-truth annotations."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to one PCB image from the annotated dataset."
    )

    parser.add_argument(
        "--padding",
        type=float,
        default=0.75,
        help="Context around each defect bounding box."
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Enlargement factor for individual defect crops."
    )

    args = parser.parse_args()

    create_inspection(
        Path(args.image),
        padding_ratio=args.padding,
        scale=args.scale
    )
