"""Train YOLO26n-OBB on the SmartPCB dataset.

Examples (run from the project root):
    python detection/train.py --name grouped_640
    python detection/train.py --name grouped_1024 --imgsz 1024 --batch 2 --nbs 4
"""
import argparse
import math
import platform
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {
    "random": ROOT / "datasets" / "processed" / "v2_obb_split",     # original random split
    "grouped": ROOT / "datasets" / "processed" / "v2_obb_grouped",  # filename-prefix groups kept together
}
RUNS_DIR = ROOT / "runs" / "train"
WEIGHTS = ROOT / "yolo26n-obb.pt"


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO26n-OBB on PCB defects")
    p.add_argument("--name", required=True, help="run folder name, e.g. grouped_640")
    p.add_argument("--split", choices=list(SPLITS), default="grouped")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--nbs", type=int, default=None,
                   help="nominal batch size; weights update every nbs/batch iterations (default: same as --batch)")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="use only this fraction of the train images (for quick smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    a = parse_args()
    torch.set_num_threads(4)

    split_dir = SPLITS[a.split]
    data_yaml = split_dir / "data.yaml"
    n_train = len(list((split_dir / "images" / "train").glob("*.jpg")))
    n_val = len(list((split_dir / "images" / "val").glob("*.jpg")))

    nbs = a.nbs or a.batch
    accumulate = max(round(nbs / a.batch), 1)
    n_used = max(round(n_train * a.fraction), 1)
    iterations = math.ceil(n_used / a.batch)
    updates = math.ceil(iterations / accumulate)
    print(f"Split: {a.split} | train images used: {n_used} of {n_train} | val images: {n_val}")
    print(f"Batch: {a.batch} | nbs: {nbs} | accumulate: {accumulate}")
    print(f"Iterations per epoch: {iterations} | weight updates per epoch: {updates}")

    model = YOLO(str(WEIGHTS))
    model.train(
        data=str(data_yaml),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        nbs=nbs,
        fraction=a.fraction,
        workers=0,
        device="cpu",
        cache=False,
        amp=False,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        seed=a.seed,
        deterministic=True,
        project=str(RUNS_DIR),
        name=a.name,
        exist_ok=False,
        plots=True,
        save=True,
        val=True,
    )

    save_dir = Path(model.trainer.save_dir)
    info = [
        f"python: {platform.python_version()}",
        f"torch: {torch.__version__}",
        f"ultralytics: {ultralytics.__version__}",
        f"split: {a.split}",
        f"dataset yaml: {data_yaml}",
        f"train images used: {n_used} of {n_train} | val images: {n_val}",
        f"batch: {a.batch} | nbs: {nbs} | updates per epoch: {updates}",
        f"settings: {vars(a)}",
    ]
    (save_dir / "run_info.txt").write_text("\n".join(info) + "\n")
    print("Run info saved to:", save_dir / "run_info.txt")


if __name__ == "__main__":
    main()
