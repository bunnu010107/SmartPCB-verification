import cv2
import argparse
from pathlib import Path


def load_image(image_path, mode="color"):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if mode == "color":
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    elif mode == "grayscale":
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    else:
        raise ValueError("Mode must be 'color' or 'grayscale'")

    if image is None:
        raise ValueError("Unable to decode image.")

    return image


def display_image_information(image):
    height, width = image.shape[:2]

    print("\n--- PCB IMAGE INFORMATION ---")
    print(f"Width: {width} pixels")
    print(f"Height: {height} pixels")
    print(f"Dimensions: {image.shape}")
    print(f"Data type: {image.dtype}")
    print(f"Image size: {image.nbytes / 1024:.2f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SmartPCB image loading utility"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the PCB image"
    )

    parser.add_argument(
        "--mode",
        choices=["color", "grayscale"],
        default="color"
    )

    args = parser.parse_args()

    pcb_image = load_image(args.image, args.mode)

    display_image_information(pcb_image)

    print("\nImage loaded successfully!")
