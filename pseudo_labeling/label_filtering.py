"""
label_filtering.py
===================
Stage 2 of pseudo-labeling: select reliable pseudo-labels from the
candidate boxes produced by `generate_pseudo_labels.py`.

Filtering rules (applied in this order)
-----------------------------------------
1. Confidence threshold  : keep boxes with conf >= `--conf_threshold`
                            (optionally overridden per class via
                            `--per_class_conf "Traffic Light=0.4,Hydrant=0.35"`).
2. Geometric sanity      : drop boxes whose normalized area is below
                            `--min_area` (tiny, usually spurious) or above
                            `--max_area` (near-full-frame, usually spurious
                            for object classes in tile imagery), and boxes
                            with non-positive width/height.
3. Duplicate suppression : within an image, class-wise greedy suppression of
                            boxes overlapping a higher-confidence box of the
                            same class with IoU > `--dedup_iou`.
4. Per-image cap         : keep at most `--max_boxes` boxes per image
                            (highest confidence first).
5. Empty-image policy    : images left with no boxes are either excluded
                            from the merged training set (default) or kept
                            as background samples (`--keep_empty`).

Final labels are written in standard 5-column YOLO format (confidence
removed) and a retention report (`filtering_report.json` + `.csv`) is
produced so every number in the paper's pseudo-label statistics table can
be regenerated.

Usage:
    python label_filtering.py \
        --raw_labels_dir /content/pseudo_labels_raw \
        --output_dir /content/pseudo_labels_filtered \
        --conf_threshold 0.5
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter

TAXONOMY_CLASSES = [
    "Bicycle", "Bridge", "Bus", "Car", "Chimney", "Crosswalk",
    "Hydrant", "Motorcycle", "Other", "Palm", "Stair", "Traffic Light",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter candidate pseudo-labels.")
    p.add_argument("--raw_labels_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./pseudo_labels_filtered")
    p.add_argument("--conf_threshold", type=float, default=0.5)
    p.add_argument("--per_class_conf", type=str, default="",
                    help="Comma-separated overrides, e.g. 'Traffic Light=0.4,Hydrant=0.35'.")
    p.add_argument("--min_area", type=float, default=0.0005, help="Min normalized box area.")
    p.add_argument("--max_area", type=float, default=0.9, help="Max normalized box area.")
    p.add_argument("--dedup_iou", type=float, default=0.8)
    p.add_argument("--max_boxes", type=int, default=20)
    p.add_argument("--keep_empty", action="store_true",
                    help="Keep images with no surviving boxes as background samples.")
    return p.parse_args()


def parse_per_class(spec: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, val = item.split("=")
        out[TAXONOMY_CLASSES.index(name.strip())] = float(val)
    return out


def iou_cxcywh(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def read_candidates(path: str):
    boxes = []
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            boxes.append((int(parts[0]), *map(float, parts[1:6])))  # (cls, xc, yc, w, h, conf)
    return boxes


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    per_class_thr = parse_per_class(args.per_class_conf)

    label_files = sorted(glob.glob(os.path.join(args.raw_labels_dir, "*.txt")))
    drop = Counter()
    kept_per_class = Counter()
    n_candidates = n_kept = n_images_kept = 0
    rows = []

    for path in label_files:
        boxes = read_candidates(path)
        n_candidates += len(boxes)

        stage = []
        for cls, xc, yc, w, h, conf in boxes:
            if conf < per_class_thr.get(cls, args.conf_threshold):
                drop["low_confidence"] += 1
                continue
            if w <= 0 or h <= 0 or w * h < args.min_area or w * h > args.max_area:
                drop["geometry"] += 1
                continue
            stage.append((cls, xc, yc, w, h, conf))

        stage.sort(key=lambda b: -b[5])
        selected = []
        for b in stage:
            if any(s[0] == b[0] and iou_cxcywh(s[1:5], b[1:5]) > args.dedup_iou for s in selected):
                drop["duplicate"] += 1
                continue
            selected.append(b)
        if len(selected) > args.max_boxes:
            drop["per_image_cap"] += len(selected) - args.max_boxes
            selected = selected[: args.max_boxes]

        name = os.path.basename(path)
        if not selected and not args.keep_empty:
            rows.append({"file": name, "candidates": len(boxes), "kept": 0, "image_kept": 0})
            continue

        with open(os.path.join(args.output_dir, name), "w") as f:
            f.write("\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h, _ in selected))
        n_kept += len(selected)
        n_images_kept += 1
        for b in selected:
            kept_per_class[TAXONOMY_CLASSES[b[0]] if b[0] < len(TAXONOMY_CLASSES) else f"class_{b[0]}"] += 1
        rows.append({"file": name, "candidates": len(boxes), "kept": len(selected), "image_kept": 1})

    report = {
        "num_label_files": len(label_files),
        "candidate_boxes": n_candidates,
        "kept_boxes": n_kept,
        "box_retention_rate": (n_kept / n_candidates) if n_candidates else None,
        "images_kept": n_images_kept,
        "image_retention_rate": (n_images_kept / len(label_files)) if label_files else None,
        "dropped_boxes_by_reason": dict(drop),
        "kept_boxes_per_class": dict(kept_per_class),
        "params": vars(args),
    }
    with open(os.path.join(args.output_dir, "filtering_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.output_dir, "filtering_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "candidates", "kept", "image_kept"])
        w.writeheader()
        w.writerows(rows)

    print(json.dumps({k: v for k, v in report.items() if k != "params"}, indent=2))
    print(f"\nFiltered labels written to {args.output_dir}")


if __name__ == "__main__":
    main()
