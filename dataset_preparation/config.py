"""
config.py
=========
Shared configuration constants for the dataset-preparation stage.

Two distinct class vocabularies are used in this project, and keeping them
explicit here avoids the ambiguity that caused several `KeyError` /
mislabeling issues during initial data exploration:

1. `TAXONOMY_CLASSES` (12 classes) — the full reCAPTCHAv2 object taxonomy,
   used as a *fallback* labeling source when an image has no corresponding
   YOLO `.txt` annotation file. In that case the image's parent-folder name
   is matched (case-insensitively) against this list to assign a single
   dominant class label.

2. `RAW_ANNOTATION_CLASS_MAP` — the class-index-to-name mapping actually
   observed inside the dataset's own YOLO-format `.txt` annotation files
   (`mikhailma/test-dataset` on Kaggle ships pre-existing bounding-box
   annotations for a 3-class subset only: Chimney, Crosswalk, Stair).
   `class_id` values encountered in the raw `.txt` files that are *not* in
   this map fall back to a generic `class_<id>` name.

Both are consumed by `build_label_dataframe.py` and `split_train_val_test.py`.
"""

# Full 12-class reCAPTCHAv2 object taxonomy (used for folder-name fallback).
TAXONOMY_CLASSES = [
    "Bicycle", "Bridge", "Bus", "Car", "Chimney", "Crosswalk",
    "Hydrant", "Motorcycle", "Other", "Palm", "Stair", "Traffic Light",
]
CLASS_TO_IDX = {c.lower(): i for i, c in enumerate(TAXONOMY_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(TAXONOMY_CLASSES)}
NUM_CLASSES = len(TAXONOMY_CLASSES)

# Class-index -> class-name mapping for the raw YOLO .txt annotations that
# ship with the mikhailma/test-dataset Kaggle dataset (3-class subset).
RAW_ANNOTATION_CLASS_MAP = {0: "Chimney", 1: "Crosswalk", 2: "Stair"}

# Kaggle dataset reference (see download_data.py).
KAGGLE_DATASET_REF = "mikhailma/test-dataset"

# Cap applied when pairing images with .txt annotations, for fast
# iteration on limited compute (Colab-scale). Set to None to process all.
MAX_LABEL_FILES = 1000

# Fraction of the dataset held out for validation.
VAL_SPLIT_RATIO = 0.2
RANDOM_SEED = 42
