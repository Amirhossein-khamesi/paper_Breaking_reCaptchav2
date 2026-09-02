"""
download_data.py
=================
Acquires the raw reCAPTCHAv2 image corpus from Kaggle
(`mikhailma/test-dataset`) via `kagglehub`, and reports basic dataset
statistics: directory structure, image count, and YOLO-format `.txt`
annotation count. This script is the entry point of the data-preparation
stage and should be run once before `build_label_dataframe.py`.

Dataset source
--------------
Kaggle: https://www.kaggle.com/datasets/mikhailma/test-dataset

Usage:
    python download_data.py
    python download_data.py --dataset_ref mikhailma/test-dataset --max_list 10
"""

from __future__ import annotations

import argparse
import glob
import os

import kagglehub

from config import KAGGLE_DATASET_REF

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")
LABEL_EXTENSION = "*.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download the reCAPTCHAv2 dataset from Kaggle.")
    p.add_argument("--dataset_ref", type=str, default=KAGGLE_DATASET_REF,
                    help="Kaggle dataset identifier, e.g. 'mikhailma/test-dataset'.")
    p.add_argument("--max_list", type=int, default=10,
                    help="Max number of files to print per directory when exploring the structure.")
    return p.parse_args()


def resolve_dataset_folder(base_path: str) -> str:
    """The Kaggle archive for this dataset unpacks into a nested
    'Google_Recaptcha_V2_Images_Dataset' directory; fall back to the
    archive root if that subfolder is not present (e.g. dataset updates)."""
    candidate = os.path.join(base_path, "Google_Recaptcha_V2_Images_Dataset")
    return candidate if os.path.exists(candidate) else base_path


def explore_directory_structure(dataset_folder: str, max_list: int = 10) -> None:
    """Print a shallow tree view of the dataset folder for sanity-checking
    the downloaded archive layout before running the pairing/labeling
    stage."""
    print("\nExploring directory structure:")
    for root, _dirs, files in os.walk(dataset_folder):
        level = root.replace(dataset_folder, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = "  " * (level + 1)
        for file in sorted(files)[:max_list]:
            print(f"{subindent}{file}")
        if len(files) > max_list:
            print(f"{subindent}... and {len(files) - max_list} more files")


def count_images_and_labels(dataset_folder: str) -> tuple[list[str], list[str]]:
    image_paths: list[str] = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(dataset_folder, "**", ext), recursive=True))

    txt_paths = glob.glob(os.path.join(dataset_folder, "**", LABEL_EXTENSION), recursive=True)

    print(f"\nFound {len(image_paths)} images.")
    print(f"Found {len(txt_paths)} label files (.txt).")
    return image_paths, txt_paths


def main() -> None:
    args = parse_args()

    print(f"Downloading dataset: {args.dataset_ref}")
    base_path = kagglehub.dataset_download(args.dataset_ref)
    print("Dataset path:", base_path)

    dataset_folder = resolve_dataset_folder(base_path)
    print("Using folder:", dataset_folder)

    explore_directory_structure(dataset_folder, max_list=args.max_list)
    count_images_and_labels(dataset_folder)

    print(f"\nDataset ready at: {dataset_folder}")
    print("Next step: python build_label_dataframe.py --dataset_folder "
          f"\"{dataset_folder}\"")


if __name__ == "__main__":
    main()
