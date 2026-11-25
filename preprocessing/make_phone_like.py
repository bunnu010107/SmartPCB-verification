"""Create phone-like degraded copies of the validation images (labels unchanged).

Every effect keeps the geometry, so the original labels stay valid.
Output: datasets/processed/phonelike/<variant>/{images,labels}/val + data.yaml
Variants: control, color, blur, noise, glare, jpeg, all

Usage (from the project root):
    python preprocessing/make_phone_like.py
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "datasets" / "processed" / "v2_obb_grouped"
DST = ROOT / "datasets" / "processed" / "phonelike"
NAMES = ["missing_pad", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
SEED = 42


def color(img, rng):
    """White-balance cast, contrast, brightness and gamma change."""
    gains = rng.uniform(0.85, 1.15, size=3)   # B, G, R channel gains
    alpha = rng.uniform(0.75, 1.2)            # contrast
    beta = rng.uniform(-25, 25)               # brightness
    gamma = rng.uniform(0.8, 1.3)
    x = img.astype(np.float32) * gains * alpha + beta
    x = 255.0 * (np.clip(x, 0, 255) / 255.0) ** gamma
    return np.clip(x, 0, 255).astype(np.uint8)


def blur(img, rng):
    """Defocus: Gaussian blur scaled to the image width (sigma 0.8-2.5 px at 2048 px wide)."""
    sigma = rng.uniform(0.8, 2.5) * img.shape[1] / 2048
    return cv2.GaussianBlur(img, (0, 0), sigma)


def noise(img, rng):
    """Sensor noise (standard deviation 3-10 grey levels)."""
    std = rng.uniform(3, 10)
    x = img.astype(np.float32) + rng.normal(0, std, img.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def glare(img, rng):
    """Vignette (darker corners) plus, half the time, a soft bright highlight."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
    x = img.astype(np.float32) * (1.0 - rng.uniform(0.1, 0.3) * r ** 2)[..., None]
    if rng.random() < 0.5:
        gx, gy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        rad = rng.uniform(0.15, 0.3) * w
        spot = np.exp(-(((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * (rad / 2) ** 2)))
        x += spot[..., None] * rng.uniform(40, 90)
    return np.clip(x, 0, 255).astype(np.uint8)


def jpeg(img, rng):
    """Strong JPEG compression (quality 55-85)."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(55, 85))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


VARIANTS = {
    "control": [],
    "color": [color],
    "blur": [blur],
    "noise": [noise],
    "glare": [glare],
    "jpeg": [jpeg],
    "all": [color, glare, blur, noise, jpeg],
}


def main():
    if DST.exists():
        sys.exit(f"{DST} already exists - refusing to overwrite it.")
    images = sorted((SRC / "images" / "val").glob("*.jpg"))

    for variant, effects in VARIANTS.items():
        out_img = DST / variant / "images" / "val"
        out_lbl = DST / variant / "labels" / "val"
        out_img.mkdir(parents=True)
        out_lbl.mkdir(parents=True)
        for i, path in enumerate(images):
            rng = np.random.default_rng([SEED, i])   # same random numbers every run
            img = cv2.imread(str(path))
            for effect in effects:
                img = effect(img, rng)
            cv2.imwrite(str(out_img / path.name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            shutil.copy2(SRC / "labels" / "val" / f"{path.stem}.txt", out_lbl / f"{path.stem}.txt")

        lines = [f"path: {DST / variant}", "", "train: images/val", "val: images/val",
                 "test: images/val", "", "names:"]
        lines += [f"  {k}: {n}" for k, n in enumerate(NAMES)]
        (DST / variant / "data.yaml").write_text("\n".join(lines) + "\n")
        print(f"{variant:8}: {len(images)} images written")


if __name__ == "__main__":
    main()
