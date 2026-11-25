from pathlib import Path
import torch
from ultralytics import YOLO

# ==========================================
# SMARTPCB - 16 IMAGE OVERFITTING EXPERIMENT
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = (
    PROJECT_ROOT
    / "datasets/processed/overfit_test/data.yaml"
)

PRETRAINED_MODEL = "yolo26n-obb.pt"

EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 1
WORKERS = 0

DEVICE = "cpu"
SEED = 42


def main():

    torch.set_num_threads(4)

    print("\n===================================")
    print("SMARTPCB OVERFITTING EXPERIMENT")
    print("===================================")

    print("Dataset:", DATASET)
    print("Model:", PRETRAINED_MODEL)
    print("Training images: 16")
    print("Validation images: Same 16 images")
    print("Epochs:", EPOCHS)
    print("Image size:", IMAGE_SIZE)
    print("Device:", DEVICE)

    # Load pretrained OBB model
    model = YOLO(PRETRAINED_MODEL)

    # Start training
    results = model.train(

        data=str(DATASET),

        epochs=EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        workers=WORKERS,

        device=DEVICE,

        pretrained=True,

        optimizer="AdamW",

        lr0=0.001,

        patience=100,

        seed=SEED,

        deterministic=True,

        cache=False,

        amp=False,

        # Disable augmentation for memorization test
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        mosaic=0.0,
        mixup=0.0,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        project=str(PROJECT_ROOT / "runs/overfit"),

        name="pcb_16_image_test",

        exist_ok=False,

        plots=True,

        save=True,

        val=True
    )

    print("\n===================================")
    print("OVERFITTING EXPERIMENT COMPLETED")
    print("===================================")

    print("Results:", results)


if __name__ == "__main__":
    main()
