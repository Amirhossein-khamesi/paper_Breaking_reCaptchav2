"""
resize_normalize.py
=====================
Resizing and pixel-normalization utilities shared across the pipeline.
Different downstream stages require different target resolutions and value
ranges, so this module centralizes the exact transforms used at each stage
rather than letting them drift independently:

| Stage | Target resolution | Value range | Rationale |
|---|---|---|---|
| CycleGAN (`../cyclegan/`) | 256x256 | [-1, 1] | Matches the generator's Tanh output and InstanceNorm-based architecture |
| YOLOv8 (`../yolov8/`) | 640x640 | [0, 1] | Ultralytics' standard input convention for YOLOv8 |
| Evaluation test-time enhancement (`../evaluation/`) | native resolution | [0, 255] (uint8) | CLAHE/unsharp masking operate directly on 8-bit images |

This script can also be run as a standalone batch pre-processing step
(resizing and normalizing a directory of raw images to disk) when a
materialized, pre-resized dataset is preferred over resizing on the fly
inside the `Dataset.__getitem__` of each stage.

Usage (batch mode):
    python resize_normalize.py \
        --input_dir /content/yolo_dataset/images/train \
        --output_dir /content/yolo_dataset_resized/images/train \
        --target_size 256 --mode cyclegan
"""

from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")


def cyclegan_transform(image_size: int = 256) -> transforms.Compose:
    """Resize + normalize to [-1, 1], matching cyclegan/dataset.py."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def yolo_transform(image_size: int = 640) -> transforms.Compose:
    """Resize + scale to [0, 1] (no mean/std normalization), matching
    Ultralytics' internal preprocessing convention for YOLOv8."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def resize_keep_aspect_pad(image: np.ndarray, target_size: int, pad_value: int = 114) -> np.ndarray:
    """Letterbox resize: scale the longer side to target_size while
    preserving aspect ratio, then pad the shorter side to a square. This
    is the resizing convention YOLO uses internally at inference time and
    is provided here for parity when preprocessing images outside of the
    Ultralytics API (e.g. for the CycleGAN or evaluation pipelines)."""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_size, target_size, image.shape[2]), pad_value, dtype=image.dtype)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch-resize and normalize a directory of images.")
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--target_size", type=int, default=256)
    p.add_argument("--mode", type=str, choices=["cyclegan", "yolo_letterbox"], default="cyclegan",
                    help="'cyclegan': simple resize to a square, saved as normalized-range PNG "
                         "preview (for inspection). 'yolo_letterbox': aspect-preserving letterbox "
                         "resize with padding, saved as-is (uint8).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, "**", ext), recursive=True))
    print(f"Found {len(image_paths)} images in {args.input_dir}")

    for img_path in tqdm(image_paths, desc=f"Resizing ({args.mode})"):
        filename = os.path.basename(img_path)
        out_path = os.path.join(args.output_dir, filename)

        if args.mode == "cyclegan":
            img = Image.open(img_path).convert("RGB")
            img = img.resize((args.target_size, args.target_size), Image.BILINEAR)
            img.save(out_path)
        else:  # yolo_letterbox
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            padded = resize_keep_aspect_pad(img_bgr, args.target_size)
            cv2.imwrite(out_path, padded)

    print(f"Done. Resized images written to {args.output_dir}")


if __name__ == "__main__":
    main()
