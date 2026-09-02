"""
augmentation.py
=================
Offline, bounding-box-aware data augmentation for the YOLO-format dataset,
implemented with `albumentations`.

Relationship to YOLOv8's built-in on-the-fly augmentation
------------------------------------------------------------
`../yolov8/train_yolov8.py` already enables Ultralytics' built-in
augmentation pipeline during training (mosaic, mixup, copy-paste, HSV
color jitter, rotation, translation, and scale jitter — configured via the
`model.train(...)` call's `mosaic`, `mixup`, `copy_paste`, `hsv_*`,
`degrees`, `translate`, `scale` arguments). That on-the-fly augmentation is
re-sampled fresh every epoch and is sufficient for most training runs.

This script provides a complementary, *offline* augmentation pass intended
specifically for:
    - materializing a fixed, inspectable augmented dataset (useful for
      ablation studies that must hold the augmented set constant across
      multiple training runs, which Ultralytics' on-the-fly augmentation
      does not allow since it re-samples every epoch), and
    - deliberately oversampling minority classes in the 12-class taxonomy
      (see `../dataset_preparation/README.md`, Section 3) before the
      pseudo-labeling stage, since several classes are represented by very
      few ground-truth-annotated images.

All geometric transforms are bounding-box-aware (`albumentations.BboxParams`
with `format="yolo"`), so YOLO-format `.txt` labels are transformed
consistently with their corresponding image.

Usage:
    python augmentation.py \
        --images_dir /content/yolo_dataset/images/train \
        --labels_dir /content/yolo_dataset/labels/train \
        --output_images_dir /content/yolo_dataset_aug/images/train \
        --output_labels_dir /content/yolo_dataset_aug/labels/train \
        --multiplier 2
"""

from __future__ import annotations

import argparse
import glob
import os

import albumentations as A
import cv2
from tqdm import tqdm


def build_augmentation_pipeline(image_size: int = 640) -> A.Compose:
    """Bounding-box-aware augmentation pipeline.

    Transform choices and rationale:
        - HorizontalFlip: reCAPTCHA object classes (vehicles, hydrants,
          traffic lights, stairs, ...) are not orientation-dependent, so a
          horizontal flip is a label-preserving augmentation.
        - RandomBrightnessContrast / HueSaturationValue: reCAPTCHA tiles are
          sourced from Street View imagery under highly variable outdoor
          lighting; photometric jitter improves robustness to illumination
          shift, complementing the CLAHE-based test-time enhancement used
          in ../evaluation/.
        - ShiftScaleRotate: mild geometric jitter (rotation limited to
          +/-10 degrees, matching the YOLOv8 training-time `degrees=10.0`
          setting for consistency) improves robustness to the slight
          camera-angle variation present in Street View source imagery.
        - GaussianBlur / ISONoise: simulates the compression artifacts and
          sensor noise present in the low-resolution reCAPTCHA tiles.
        - RandomResizedCrop: encourages scale invariance, particularly
          relevant for small-object classes (e.g. Traffic Light, Hydrant).
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=15, p=0.4),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5,
                                border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114)),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
            ], p=0.3),
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.3),
            A.Resize(image_size, image_size),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
    )


def load_yolo_labels(label_path: str) -> tuple[list[list[float]], list[int]]:
    bboxes, class_labels = [], []
    if not os.path.exists(label_path):
        return bboxes, class_labels
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            bboxes.append([cx, cy, w, h])
            class_labels.append(cls_id)
    return bboxes, class_labels


def save_yolo_labels(label_path: str, bboxes: list, class_labels: list) -> None:
    with open(label_path, "w") as f:
        for (cx, cy, w, h), cls_id in zip(bboxes, class_labels):
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline bounding-box-aware augmentation for the YOLO dataset.")
    p.add_argument("--images_dir", type=str, required=True)
    p.add_argument("--labels_dir", type=str, required=True)
    p.add_argument("--output_images_dir", type=str, required=True)
    p.add_argument("--output_labels_dir", type=str, required=True)
    p.add_argument("--image_size", type=int, default=640)
    p.add_argument("--multiplier", type=int, default=2,
                    help="Number of augmented variants generated per source image.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_images_dir, exist_ok=True)
    os.makedirs(args.output_labels_dir, exist_ok=True)

    import random
    random.seed(args.seed)

    pipeline = build_augmentation_pipeline(args.image_size)

    image_paths = sorted(
        glob.glob(os.path.join(args.images_dir, "*.jpg"))
        + glob.glob(os.path.join(args.images_dir, "*.jpeg"))
        + glob.glob(os.path.join(args.images_dir, "*.png"))
    )
    print(f"Found {len(image_paths)} source images. Generating {args.multiplier} "
          f"augmented variant(s) each -> {len(image_paths) * args.multiplier} new images.")

    written, skipped_empty = 0, 0
    for img_path in tqdm(image_paths, desc="Augmenting"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(args.labels_dir, base_name + ".txt")

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        bboxes, class_labels = load_yolo_labels(label_path)

        for variant in range(args.multiplier):
            try:
                if bboxes:
                    transformed = pipeline(image=image_rgb, bboxes=bboxes, class_labels=class_labels)
                    out_bboxes, out_labels = transformed["bboxes"], transformed["class_labels"]
                else:
                    # No ground-truth boxes: apply image-only transforms (skip bbox_params requirement).
                    transformed = pipeline(image=image_rgb, bboxes=[], class_labels=[])
                    out_bboxes, out_labels = [], []
            except Exception as e:  # noqa: BLE001 - log and skip malformed samples
                print(f"Skipping {img_path} (variant {variant}) due to augmentation error: {e}")
                continue

            out_name = f"{base_name}_aug{variant}"
            out_img_path = os.path.join(args.output_images_dir, out_name + ".png")
            out_lbl_path = os.path.join(args.output_labels_dir, out_name + ".txt")

            out_image_bgr = cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_img_path, out_image_bgr)
            save_yolo_labels(out_lbl_path, out_bboxes, out_labels)

            written += 1
            if not out_bboxes:
                skipped_empty += 1

    print(f"\nDone. Wrote {written} augmented image/label pairs to "
          f"{args.output_images_dir} / {args.output_labels_dir}")
    print(f"({skipped_empty} augmented images have no ground-truth boxes, "
          f"consistent with their unannotated source image.)")


if __name__ == "__main__":
    main()
