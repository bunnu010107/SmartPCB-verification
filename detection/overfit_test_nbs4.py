import torch
from ultralytics import YOLO

torch.set_num_threads(4)

model = YOLO("yolo26n-obb.pt")

model.train(
    data="datasets/processed/overfit_test/data.yaml",
    epochs=100,
    imgsz=640,
    batch=4,
    nbs=4,
    workers=0,
    device="cpu",
    cache=False,
    amp=False,
    pretrained=True,
    optimizer="AdamW",
    lr0=0.001,
    degrees=0,
    translate=0,
    scale=0,
    shear=0,
    perspective=0,
    mosaic=0,
    mixup=0,
    fliplr=0,
    flipud=0,
    hsv_h=0,
    hsv_s=0,
    hsv_v=0,
    seed=42,
    deterministic=True,
    patience=100,
    project="runs/overfit",
    name="pcb_16_image_nbs4",
    exist_ok=False,
    plots=True,
    save=True,
    val=True,
)
