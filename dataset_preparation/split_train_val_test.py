"""
split_train_val_test.py
========================
Performs a stratified train/validation split of the labeled dataset
produced by `build_label_dataframe.py`, materializes the split into the
YOLO-expected directory layout, and writes the `dataset.yaml` configuration
file consumed by `yolov8/train_yolov8.py`.

Output layout:
    <output_dir>/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── dataset.yaml

Handling of images without ground-truth bounding boxes
--------------------------------------------------------
Images that were only folder-name-labeled (no `.txt` annotation available;
see `build_label_dataframe.py`) are still copied into the YOLO image split,
but receive an *empty* `.txt` label file, consistent with the YOLO
convention for background/unannotated images. This preserves dataset size
for the CycleGAN enhancement and pseudo-labeling stages (which do not
require ground-truth boxes) while remaining a valid YOLO training layout.

Usage:
    python split_train_val_test.py --labels_csv labels_df.csv --output_dir /content/yolo_dataset
"""

from __future__ import annotations

import argparse
import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_SEED, VAL_SPLIT_RATIO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split the labeled dataset and build the YOLO directory layout.")
    p.add_argument("--labels_csv", type=str, default="labels_df.csv",
                    help="Output of build_label_dataframe.py.")
    p.add_argument("--output_dir", type=str, default="./yolo_dataset")
    p.add_argument("--val_ratio", type=float, default=VAL_SPLIT_RATIO)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--min_samples_for_split", type=int, default=5,
                    help="Below this many total samples, skip splitting and put everything in train.")
    return p.parse_args()


def stratified_split(labels_df: pd.DataFrame, val_ratio: float, seed: int, min_samples: int):
    if len(labels_df) < min_samples:
        print(f"Only {len(labels_df)} samples available (< {min_samples}) "
              f"-> all samples assigned to train, val left empty.")
        return labels_df.copy(), labels_df.iloc[0:0].copy()

    try:
        train_df, val_df = train_test_split(
            labels_df, test_size=val_ratio, stratify=labels_df["label"], random_state=seed,
        )
        print(f"Stratified split succeeded -> train: {len(train_df)}, val: {len(val_df)}")
    except ValueError:
        # Falls back to a non-stratified split when a class has too few
        # samples for stratification (scikit-learn requires >= 2 per class).
        print("Stratified split not possible (a class has <2 samples) "
              "-> falling back to a random split.")
        train_df, val_df = train_test_split(labels_df, test_size=val_ratio, random_state=seed)
        print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    return train_df, val_df


def materialize_yolo_layout(train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: str):
    for split in ("train", "val"):
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    print("Copying images and labels into the YOLO directory layout...")
    copied_with_labels, created_empty = 0, 0

    for split, df in (("train", train_df), ("val", val_df)):
        for _, row in df.iterrows():
            dst_img = os.path.join(output_dir, "images", split, row["filename"])
            shutil.copy(row["img_path"], dst_img)

            txt_path = row.get("txt_path")
            if pd.notna(txt_path) and txt_path:
                dst_txt = os.path.join(output_dir, "labels", split, os.path.basename(txt_path))
                shutil.copy(txt_path, dst_txt)
                copied_with_labels += 1
            else:
                txt_filename = row["filename"].rsplit(".", 1)[0] + ".txt"
                open(os.path.join(output_dir, "labels", split, txt_filename), "w").close()
                created_empty += 1

    print("Done.")
    print(f"  - Real YOLO label files copied: {copied_with_labels}")
    print(f"  - Empty .txt files created (unannotated images): {created_empty}")
    return copied_with_labels, created_empty


def infer_detection_classes(train_df: pd.DataFrame, val_df: pd.DataFrame) -> list[str]:
    """Derive the class list to write into dataset.yaml from the actual
    ground-truth .txt annotations present in the split (i.e. the classes
    the detector will actually be supervised on), not the full 12-class
    folder taxonomy used for stratification bookkeeping."""
    from config import RAW_ANNOTATION_CLASS_MAP

    detection_class_ids: set[int] = set()
    for df in (train_df, val_df):
        for txt_path in df["txt_path"]:
            if pd.isna(txt_path) or not txt_path:
                continue
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        detection_class_ids.add(int(parts[0]))

    if not detection_class_ids:
        print("No ground-truth detection classes found -> defaulting to "
              "the 3-class annotated subset (Chimney, Crosswalk, Stair).")
        return ["Chimney", "Crosswalk", "Stair"]

    return [RAW_ANNOTATION_CLASS_MAP.get(cid, f"class_{cid}") for cid in sorted(detection_class_ids)]


def write_dataset_yaml(output_dir: str, class_names: list[str]) -> str:
    yaml_content = (
        f"path: {output_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\nWrote dataset.yaml:\n{yaml_content}")
    return yaml_path


def main() -> None:
    args = parse_args()
    labels_df = pd.read_csv(args.labels_csv)
    print(f"Loaded {len(labels_df)} labeled samples from {args.labels_csv}")

    train_df, val_df = stratified_split(labels_df, args.val_ratio, args.seed, args.min_samples_for_split)
    materialize_yolo_layout(train_df, val_df, args.output_dir)

    class_names = infer_detection_classes(train_df, val_df)
    write_dataset_yaml(args.output_dir, class_names)

    print(f"\nYOLO-format dataset ready at: {args.output_dir}")
    print("Next step: python ../preprocessing/resize_normalize.py "
          f"--data_dir {args.output_dir}")


if __name__ == "__main__":
    main()
