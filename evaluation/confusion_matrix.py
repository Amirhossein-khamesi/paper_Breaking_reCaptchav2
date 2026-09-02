"""
confusion_matrix.py
=====================
Builds an IoU-matched multi-class confusion matrix (including an explicit
"background" row/column for false positives and false negatives) from the
trained detector's predictions on the validation split, and renders it as
a publication-ready heatmap figure.

This complements the confusion matrix Ultralytics generates internally
during `model.val()` (see `compute_metrics.py`) with a version whose
matching logic (greedy IoU-based assignment, configurable IoU threshold)
is fully specified here and therefore directly reproducible and citable in
the paper's Methodology / Evaluation section, rather than relying on an
external library's internal (and less-documented) implementation.

Matching procedure
-------------------
For each image:
    1. Predicted boxes are sorted by descending confidence.
    2. Each predicted box is greedily matched to the highest-IoU
       unmatched ground-truth box of *any* class, provided IoU >= --iou_thr.
    3. A matched pair contributes to cell (gt_class, pred_class) of the
       confusion matrix (on-diagonal if correctly classified).
    4. An unmatched prediction (no ground-truth box with sufficient IoU)
       is counted as a false positive: cell (background, pred_class).
    5. An unmatched ground-truth box (no prediction with sufficient IoU)
       is counted as a false negative: cell (gt_class, background).

Usage:
    python confusion_matrix.py \
        --weights /content/yolo_runs/final_with_cyclegan/weights/best.pt \
        --images_dir /content/yolo_dataset/images/val \
        --labels_dir /content/yolo_dataset/labels/val \
        --class_names Chimney Crosswalk Stair \
        --output_dir ./results
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

BACKGROUND_LABEL = "background"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an IoU-matched confusion matrix on the validation split.")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--images_dir", type=str, required=True)
    p.add_argument("--labels_dir", type=str, required=True,
                    help="Directory of YOLO-format .txt ground-truth files matching images_dir.")
    p.add_argument("--class_names", type=str, nargs="+", required=True,
                    help="Ordered class names matching the ground-truth class indices, "
                         "e.g. --class_names Chimney Crosswalk Stair")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou_thr", type=float, default=0.5,
                    help="Minimum IoU for a prediction to be matched to a ground-truth box.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--output_dir", type=str, default="./results")
    return p.parse_args()


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def load_ground_truth(label_path: str, img_w: int, img_h: int):
    boxes, classes = [], []
    if not os.path.exists(label_path):
        return boxes, classes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append(yolo_to_xyxy(cx, cy, w, h, img_w, img_h))
            classes.append(cls_id)
    return boxes, classes


def iou_xyxy(box_a, box_b) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def match_predictions_to_gt(pred_boxes, pred_classes, pred_confs, gt_boxes, gt_classes, iou_thr: float):
    """Greedy IoU matching, sorted by prediction confidence (descending).
    Returns a list of (gt_class_or_None, pred_class_or_None) pairs."""
    order = np.argsort(-np.array(pred_confs)) if pred_confs else []
    matched_gt = set()
    pairs = []

    for idx in order:
        p_box, p_cls = pred_boxes[idx], pred_classes[idx]
        best_iou, best_j = 0.0, -1
        for j, g_box in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = iou_xyxy(p_box, g_box)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            matched_gt.add(best_j)
            pairs.append((gt_classes[best_j], p_cls))
        else:
            pairs.append((None, p_cls))  # false positive

    for j in range(len(gt_boxes)):
        if j not in matched_gt:
            pairs.append((gt_classes[j], None))  # false negative

    return pairs


def build_confusion_matrix(pairs, class_names: list[str]) -> np.ndarray:
    n = len(class_names) + 1  # + background
    bg_idx = n - 1
    matrix = np.zeros((n, n), dtype=int)
    for gt_cls, pred_cls in pairs:
        row = gt_cls if gt_cls is not None else bg_idx
        col = pred_cls if pred_cls is not None else bg_idx
        if row < n and col < n:
            matrix[row, col] += 1
    return matrix


def plot_confusion_matrix(matrix: np.ndarray, labels: list[str], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 3, 1.1 * len(labels) + 3))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title("IoU-Matched Confusion Matrix (IoU >= threshold)")

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero for empty rows
    norm = matrix / row_sums

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = YOLO(args.weights)
    class_names = args.class_names
    labels_with_bg = class_names + [BACKGROUND_LABEL]

    image_paths = sorted(
        glob.glob(os.path.join(args.images_dir, "*.jpg"))
        + glob.glob(os.path.join(args.images_dir, "*.jpeg"))
        + glob.glob(os.path.join(args.images_dir, "*.png"))
    )
    print(f"Evaluating {len(image_paths)} validation images "
          f"(conf >= {args.conf}, IoU >= {args.iou_thr})...")

    all_pairs = []
    for img_path in image_paths:
        results = model.predict(img_path, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        img_h, img_w = results.orig_shape

        pred_boxes, pred_classes, pred_confs = [], [], []
        if results.boxes is not None:
            for box in results.boxes:
                pred_boxes.append(box.xyxy[0].cpu().numpy().tolist())
                pred_classes.append(int(box.cls.item()))
                pred_confs.append(float(box.conf.item()))

        label_path = os.path.join(
            args.labels_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        )
        gt_boxes, gt_classes = load_ground_truth(label_path, img_w, img_h)

        pairs = match_predictions_to_gt(pred_boxes, pred_classes, pred_confs, gt_boxes, gt_classes, args.iou_thr)
        all_pairs.extend(pairs)

    matrix = build_confusion_matrix(all_pairs, class_names)

    csv_path = os.path.join(args.output_dir, "confusion_matrix.csv")
    with open(csv_path, "w") as f:
        f.write("," + ",".join(labels_with_bg) + "\n")
        for i, row_label in enumerate(labels_with_bg):
            f.write(row_label + "," + ",".join(str(v) for v in matrix[i]) + "\n")
    print(f"Saved confusion matrix table to {csv_path}")

    fig_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(matrix, labels_with_bg, fig_path)
    print(f"Saved confusion matrix figure to {fig_path}")

    total_gt = sum(1 for gt, _ in all_pairs if gt is not None)
    total_pred = sum(1 for _, pred in all_pairs if pred is not None)
    correct = sum(1 for gt, pred in all_pairs if gt is not None and pred is not None and gt == pred)
    print(f"\nGround-truth boxes: {total_gt} | Predicted boxes: {total_pred} | "
          f"Correctly classified matches: {correct}")


if __name__ == "__main__":
    main()
