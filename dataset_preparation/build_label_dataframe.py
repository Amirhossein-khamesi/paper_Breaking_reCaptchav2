"""
build_label_dataframe.py
=========================
Pairs raw images with their corresponding YOLO-format `.txt` annotation
files (when present), assigns a single dominant class label per image for
dataset-level bookkeeping and stratified splitting, and falls back to a
folder-name heuristic for images that have no annotation file.

Why a two-stage labeling strategy is needed
--------------------------------------------
The raw `mikhailma/test-dataset` archive contains YOLO-format bounding-box
annotations for only a 3-class subset of the full 12-class reCAPTCHAv2
taxonomy (`Chimney`, `Crosswalk`, `Stair` — see `config.RAW_ANNOTATION_CLASS_MAP`).
The remaining images are organized into class-named subdirectories without
per-image annotation files. To (a) retain every usable image for training,
and (b) still be able to perform a stratified train/validation split by
class, we:

    1. Pair every `.txt` file with its corresponding image by filename,
       and assign each paired image a single dominant label — the class of
       the *first* bounding box listed in its annotation file.
    2. For any remaining unpaired image, infer its class from its
       parent-directory name (case-insensitive substring match against the
       12-class taxonomy), defaulting to `"Other"` if no match is found.
    3. If step 1 recovers labels for fewer than 10% of all discovered
       images (i.e. the `.txt`-annotated subset is too small to be
       representative), step 2 is additionally applied to *all* images,
       ensuring the working dataset is never bottlenecked by annotation
       coverage.

This produces a single `labels_df` (pandas DataFrame) with columns
`filename, label, label_idx, img_path, txt_path`, saved to
`labels_df.csv`, which is the direct input to `split_train_val_test.py`.

Usage:
    python build_label_dataframe.py --dataset_folder /path/to/dataset_folder
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd
from tqdm import tqdm

from config import CLASS_TO_IDX, MAX_LABEL_FILES, RAW_ANNOTATION_CLASS_MAP, TAXONOMY_CLASSES

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pair images with YOLO labels and build the label dataframe.")
    p.add_argument("--dataset_folder", type=str, required=True,
                    help="Root folder of the downloaded dataset (output of download_data.py).")
    p.add_argument("--output_csv", type=str, default="labels_df.csv")
    p.add_argument("--max_label_files", type=int, default=MAX_LABEL_FILES,
                    help="Cap on the number of .txt files scanned for image-label pairing.")
    p.add_argument("--fallback_threshold", type=float, default=0.10,
                    help="If the fraction of images successfully paired with .txt labels falls "
                         "below this threshold, folder-name fallback labeling is applied to the "
                         "full image set, not just the unpaired subset.")
    return p.parse_args()


def list_files(root: str, patterns) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    return paths


def pair_images_with_labels(dataset_folder: str, txt_paths: list[str]) -> list[tuple[str, str]]:
    """Pair each .txt annotation file with the image of the same base name,
    searching first in the same directory, then (as a fallback) anywhere
    under dataset_folder."""
    pairs: list[tuple[str, str]] = []
    for txt_path in tqdm(txt_paths, desc="Pairing labels"):
        base_no_ext = os.path.basename(txt_path).replace(".txt", "")
        dir_path = os.path.dirname(txt_path)

        img_path = None
        for ext in IMAGE_EXTENSIONS:
            candidate = os.path.join(dir_path, base_no_ext + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            for ext in IMAGE_EXTENSIONS:
                matches = glob.glob(os.path.join(dataset_folder, "**", base_no_ext + ext), recursive=True)
                if matches:
                    img_path = matches[0]
                    break
        if img_path is not None:
            pairs.append((img_path, txt_path))
    return pairs


def dominant_label_from_annotation(txt_path: str) -> tuple[str, list[int]]:
    """Return the dominant class name (class of the first bounding box) and
    the full list of class ids present in a YOLO-format .txt annotation
    file. An empty file yields ('Other'-mapped id 0, [])."""
    with open(txt_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    class_ids = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            class_ids.append(int(parts[0]))

    main_class_id = class_ids[0] if class_ids else 0
    main_label = RAW_ANNOTATION_CLASS_MAP.get(main_class_id, "Other")
    return main_label, class_ids


def label_from_folder_name(img_path: str) -> str:
    """Infer a class label from the image's parent-directory name via
    case-insensitive substring matching against the full taxonomy."""
    dir_name = os.path.basename(os.path.dirname(img_path)).lower()
    for cls in TAXONOMY_CLASSES:
        if cls.lower() in dir_name:
            return cls
    return "Other"


def build_dataframe(dataset_folder: str, max_label_files: int, fallback_threshold: float) -> pd.DataFrame:
    image_paths = list_files(dataset_folder, ("*.jpg", "*.jpeg", "*.png"))
    txt_paths = list_files(dataset_folder, ("*.txt",))
    print(f"Found {len(image_paths)} images and {len(txt_paths)} label files.")

    limited_txt_paths = txt_paths[:max_label_files] if max_label_files else txt_paths
    pairs = pair_images_with_labels(dataset_folder, limited_txt_paths)
    print(f"Paired {len(pairs)} image-label pairs (scanned {len(limited_txt_paths)} .txt files).")

    records = []
    detection_class_ids: set[int] = set()
    for img_path, txt_path in pairs:
        main_label, class_ids = dominant_label_from_annotation(txt_path)
        detection_class_ids.update(class_ids)
        records.append({
            "filename": os.path.basename(img_path),
            "label": main_label,
            "label_idx": CLASS_TO_IDX.get(main_label.lower(), len(TAXONOMY_CLASSES) - 1),
            "img_path": img_path,
            "txt_path": txt_path,
        })

    labels_df = pd.DataFrame(records)
    print(f"Labeled samples from .txt annotations: {len(labels_df)}")

    coverage = (len(pairs) / len(image_paths)) if image_paths else 0.0
    if coverage < fallback_threshold or len(labels_df) == 0:
        print(f"Annotation coverage ({coverage:.1%}) below threshold "
              f"({fallback_threshold:.1%}) -> applying folder-name fallback "
              f"labeling to unpaired images.")
        existing_filenames = set(labels_df["filename"].tolist()) if len(labels_df) else set()
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            if filename in existing_filenames:
                continue
            matched_label = label_from_folder_name(img_path)
            records.append({
                "filename": filename,
                "label": matched_label,
                "label_idx": CLASS_TO_IDX.get(matched_label.lower(), len(TAXONOMY_CLASSES) - 1),
                "img_path": img_path,
                "txt_path": None,
            })
            existing_filenames.add(filename)
        labels_df = pd.DataFrame(records)

    print("\nFinal class distribution:")
    print(labels_df["label"].value_counts())
    return labels_df


def main() -> None:
    args = parse_args()
    labels_df = build_dataframe(args.dataset_folder, args.max_label_files, args.fallback_threshold)
    labels_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved label dataframe ({len(labels_df)} rows) to {args.output_csv}")
    print("Next step: python split_train_val_test.py --labels_csv "
          f"{args.output_csv}")


if __name__ == "__main__":
    main()
