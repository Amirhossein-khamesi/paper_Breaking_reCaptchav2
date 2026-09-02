"""
compute_metrics.py
====================
Primary quantitative evaluation of the trained YOLOv8 detector: mAP@50,
mAP@50-95, precision, and recall, computed via Ultralytics' native
validation routine (`YOLO.val`), plus per-class breakdowns and a
machine-readable results table (CSV + JSON) for direct inclusion in the
paper.

This is the evaluation counterpart of `../yolov8/train_yolov8.py`; it must
be run against the same `dataset.yaml` used for training so that class
indices and the validation split are consistent.

Metrics reported (standard COCO/PASCAL-VOC-style object-detection metrics):
    - mAP@0.5        : mean Average Precision at IoU threshold 0.5
    - mAP@0.5:0.95    : mean Average Precision averaged over IoU
                        thresholds 0.5-0.95 (step 0.05), the primary COCO metric
    - Precision       : box precision, macro-averaged over classes
    - Recall          : box recall, macro-averaged over classes
    - Per-class AP@0.5 and AP@0.5:0.95

Handling of unannotated validation images
-------------------------------------------
As documented in `../dataset_preparation/README.md`, a subset of images in
this dataset only carry folder-inferred category labels and have empty
YOLO `.txt` ground-truth files (no bounding boxes). When the validation
split consists partly or entirely of such images, precision/recall/mAP are
not well-defined (there is no ground truth to compare against) and are
reported as `N/A` rather than a misleading `0.0000`, with an explicit note
printed to the console and written into the results file.

Usage:
    python compute_metrics.py \
        --weights /content/yolo_runs/final_with_cyclegan/weights/best.pt \
        --data /content/yolo_dataset/dataset.yaml \
        --output_dir ./results
"""

from __future__ import annotations

import argparse
import json
import os

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute standard YOLOv8 validation metrics.")
    p.add_argument("--weights", type=str, required=True,
                    help="Path to the trained model checkpoint (best.pt or last.pt).")
    p.add_argument("--data", type=str, required=True,
                    help="Path to dataset.yaml (must match the training configuration).")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default=None,
                    help="'cuda', 'cpu', or a specific device index. Defaults to auto-detect.")
    p.add_argument("--output_dir", type=str, default="./results")
    p.add_argument("--plots", action="store_true", default=True,
                    help="Generate Ultralytics' built-in PR-curve / confusion-matrix plots.")
    return p.parse_args()


def has_valid_array(arr) -> bool:
    return arr is not None and len(arr) > 0


def summarize_metrics(metrics, class_names: dict[int, str]) -> dict:
    """Extract a JSON-serializable summary from an Ultralytics DetMetrics
    object, guarding against the case where the validation split has no
    ground-truth boxes (see module docstring)."""
    summary: dict = {
        "map50": float(metrics.box.map50) if metrics.box.map50 is not None else None,
        "map50_95": float(metrics.box.map) if metrics.box.map is not None else None,
    }

    if has_valid_array(getattr(metrics.box, "p", None)):
        summary["precision_macro"] = float(metrics.box.p.mean())
    else:
        summary["precision_macro"] = None

    if has_valid_array(getattr(metrics.box, "r", None)):
        summary["recall_macro"] = float(metrics.box.r.mean())
    else:
        summary["recall_macro"] = None

    # Per-class AP, when available.
    per_class = {}
    if has_valid_array(getattr(metrics.box, "ap50", None)) and has_valid_array(getattr(metrics.box, "ap", None)):
        ap_class_indices = getattr(metrics.box, "ap_class_index", range(len(metrics.box.ap50)))
        for i, cls_idx in enumerate(ap_class_indices):
            cls_name = class_names.get(int(cls_idx), f"class_{cls_idx}")
            per_class[cls_name] = {
                "AP50": float(metrics.box.ap50[i]),
                "AP50_95": float(metrics.box.ap[i]),
            }
    summary["per_class_AP"] = per_class

    summary["has_ground_truth"] = bool(per_class) or summary["precision_macro"] is not None
    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("Final Evaluation Results")
    print("=" * 60)
    map50 = summary["map50"]
    map95 = summary["map50_95"]
    print(f"mAP@0.5      : {map50:.4f}" if map50 is not None else "mAP@0.5      : N/A")
    print(f"mAP@0.5:0.95 : {map95:.4f}" if map95 is not None else "mAP@0.5:0.95 : N/A")

    if summary["precision_macro"] is not None:
        print(f"Precision (macro): {summary['precision_macro']:.4f}")
    else:
        print("Precision (macro): N/A (no ground-truth boxes in the validation split)")

    if summary["recall_macro"] is not None:
        print(f"Recall (macro):    {summary['recall_macro']:.4f}")
    else:
        print("Recall (macro):    N/A (no ground-truth boxes in the validation split)")

    if summary["per_class_AP"]:
        print("\nPer-class AP:")
        print(f"{'Class':<20}{'AP@0.5':>10}{'AP@0.5:0.95':>15}")
        for cls_name, vals in summary["per_class_AP"].items():
            print(f"{cls_name:<20}{vals['AP50']:>10.4f}{vals['AP50_95']:>15.4f}")

    if not summary["has_ground_truth"]:
        print(
            "\nNote: the validation split contains no ground-truth bounding "
            "boxes for a subset (or all) of its images (see "
            "../dataset_preparation/README.md, Section 4). Precision/Recall/"
            "per-class AP are undefined in that case and reported as N/A "
            "rather than a misleading 0.0000."
        )
    print("=" * 60)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = YOLO(args.weights)
    print(f"Loaded model: {args.weights}")
    print(f"Validating against: {args.data}")

    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=args.plots,
        project=args.output_dir,
        name="val",
        exist_ok=True,
        verbose=False,
    )

    summary = summarize_metrics(metrics, model.names)
    print_summary(summary)

    json_path = os.path.join(args.output_dir, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved machine-readable summary to {json_path}")

    if summary["per_class_AP"]:
        csv_path = os.path.join(args.output_dir, "per_class_ap.csv")
        with open(csv_path, "w") as f:
            f.write("class,AP50,AP50_95\n")
            for cls_name, vals in summary["per_class_AP"].items():
                f.write(f"{cls_name},{vals['AP50']:.6f},{vals['AP50_95']:.6f}\n")
        print(f"Saved per-class AP table to {csv_path}")

    print(f"\nUltralytics validation plots (PR curve, confusion matrix, etc.) "
          f"saved under: {os.path.join(args.output_dir, 'val')}")


if __name__ == "__main__":
    main()
