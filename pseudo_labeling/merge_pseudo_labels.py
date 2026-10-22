"""
merge_pseudo_labels.py
========================
Stage 3 of pseudo-labeling: copy CycleGAN-enhanced images that survived
filtering, together with their filtered labels, into the YOLO training
split (`<dataset_dir>/images/train`, `<dataset_dir>/labels/train`).

Only images that have a filtered label file are merged, so the set of
pseudo-labeled training samples is exactly the set described by
`filtering_report.json`. Merged files get a `pseudo_` filename prefix so
that pseudo-labeled samples remain distinguishable from real samples (this
makes "real-only" vs "real + pseudo" ablations a simple filename filter).

If the validation split is empty (a situation encountered in the original
experiments, see ../dataset_preparation/README.md), `--ensure_val` moves a
fraction of the *real* (non-pseudo) training images to validation, so that
validation never contains pseudo-labeled data.

Usage:
    python merge_pseudo_labels.py \
        --images_dir /content/synthetic_enhanced \
        --filtered_labels_dir /content/pseudo_labels_filtered \
        --dataset_dir /content/yolo_dataset --ensure_val
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import shutil

PREFIX = "pseudo_"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge pseudo-labeled images into the YOLO train split.")
    p.add_argument("--images_dir", type=str, required=True)
    p.add_argument("--filtered_labels_dir", type=str, required=True)
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--ensure_val", action="store_true")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def find_image(images_dir: str, stem: str):
    for ext in (".png", ".jpg", ".jpeg"):
        cand = os.path.join(images_dir, stem + ext)
        if os.path.exists(cand):
            return cand
    return None


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    tr_img = os.path.join(args.dataset_dir, "images", "train")
    tr_lbl = os.path.join(args.dataset_dir, "labels", "train")
    va_img = os.path.join(args.dataset_dir, "images", "val")
    va_lbl = os.path.join(args.dataset_dir, "labels", "val")
    for d in (tr_img, tr_lbl, va_img, va_lbl):
        os.makedirs(d, exist_ok=True)

    merged = 0
    for lbl_path in sorted(glob.glob(os.path.join(args.filtered_labels_dir, "*.txt"))):
        stem = os.path.splitext(os.path.basename(lbl_path))[0]
        img_path = find_image(args.images_dir, stem)
        if img_path is None:
            continue
        ext = os.path.splitext(img_path)[1]
        shutil.copy(img_path, os.path.join(tr_img, PREFIX + stem + ext))
        shutil.copy(lbl_path, os.path.join(tr_lbl, PREFIX + stem + ".txt"))
        merged += 1
    print(f"Merged {merged} pseudo-labeled images into {tr_img}")

    if args.ensure_val and not glob.glob(os.path.join(va_img, "*.*")):
        real = [p for p in glob.glob(os.path.join(tr_img, "*.*"))
                if not os.path.basename(p).startswith(PREFIX)]
        n_move = max(5, int(len(real) * args.val_fraction)) if len(real) > 10 else 0
        for img_p in random.sample(real, min(n_move, len(real))):
            fn = os.path.basename(img_p)
            lbl_p = os.path.join(tr_lbl, os.path.splitext(fn)[0] + ".txt")
            shutil.move(img_p, os.path.join(va_img, fn))
            if os.path.exists(lbl_p):
                shutil.move(lbl_p, os.path.join(va_lbl, os.path.basename(lbl_p)))
        print(f"Validation split was empty -> moved {n_move} real images to val")


if __name__ == "__main__":
    main()
