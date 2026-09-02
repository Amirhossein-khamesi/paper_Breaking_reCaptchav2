"""
confidence_analysis.py
========================
Qualitative and per-class confidence analysis of the trained detector under
test-time image enhancement (CLAHE + unsharp masking, see `enhancement.py`).

For each validation (or fallback: sample of training) image, this script:
    1. applies the enhancement transform,
    2. runs detection at a high confidence threshold (default 0.80) to
       isolate the model's most reliable predictions,
    3. maps the detector's output classes onto the dataset's class
       vocabulary via `--class_map` (COCO-pretrained models predict COCO
       class names, which are remapped to the project's reCAPTCHAv2
       taxonomy — see the default mapping below),
    4. aggregates the mean detection confidence per class, and
    5. renders a grid figure of the top-K highest-confidence detections
       with bounding boxes and labels overlaid, for direct use as a
       qualitative results figure in the paper.

This script is intended as a *complement* to the quantitative metrics in
`compute_metrics.py`: it surfaces where the model is most confident and
what it looks like when it is, which is standard practice in Q1-level
detection papers alongside the numeric mAP/precision/recall table.

Usage:
    python confidence_analysis.py \
        --weights /content/yolo_runs/final_with_cyclegan/weights/best.pt \
        --images_dir /content/yolo_dataset/images/val \
        --output_dir ./results \
        --conf_threshold 0.80 --top_k 10
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

from enhancement import clahe_unsharp_enhance

# Default mapping from COCO-pretrained class names to the project's
# reCAPTCHAv2 class taxonomy (see ../dataset_preparation/config.py).
DEFAULT_CLASS_MAP = {
    "person": "Other",
    "bicycle": "Bicycle",
    "car": "Car",
    "motorcycle": "Motorcycle",
    "bus": "Bus",
    "traffic light": "Traffic Light",
    "fire hydrant": "Hydrant",
}

BOX_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 0), (255, 192, 203),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-class confidence analysis with test-time enhancement.")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--images_dir", type=str, required=True)
    p.add_argument("--fallback_images_dir", type=str, default=None,
                    help="Used if --images_dir is empty (e.g. fall back to a train sample).")
    p.add_argument("--fallback_limit", type=int, default=50)
    p.add_argument("--output_dir", type=str, default="./results")
    p.add_argument("--conf_threshold", type=float, default=0.80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--class_map_json", type=str, default=None,
                    help="Optional path to a JSON file overriding DEFAULT_CLASS_MAP "
                         "({'coco_name': 'project_class_name', ...}).")
    return p.parse_args()


def load_class_map(class_map_json: str | None) -> dict:
    if class_map_json is None:
        return DEFAULT_CLASS_MAP
    with open(class_map_json, "r") as f:
        return json.load(f)


def collect_images(images_dir: str, fallback_dir: str | None, fallback_limit: int) -> list[str]:
    images = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.jpeg"))
        + glob.glob(os.path.join(images_dir, "*.png"))
    )
    if not images and fallback_dir:
        print(f"No images found in {images_dir}; falling back to {fallback_dir} "
              f"(first {fallback_limit} images).")
        images = sorted(
            glob.glob(os.path.join(fallback_dir, "*.jpg"))
            + glob.glob(os.path.join(fallback_dir, "*.jpeg"))
            + glob.glob(os.path.join(fallback_dir, "*.png"))
        )[:fallback_limit]
    return images


def run_confidence_analysis(
    model: YOLO,
    image_paths: list[str],
    class_map: dict,
    conf_threshold: float,
    imgsz: int,
    output_dir: str,
) -> tuple[list[tuple], dict]:
    all_detections = []  # (confidence, project_class, image_path, enhanced_path, coco_class)
    class_confidences = {cls: [] for cls in set(class_map.values())}

    enhanced_dir = os.path.join(output_dir, "enhanced_images")
    os.makedirs(enhanced_dir, exist_ok=True)

    print(f"Processing {len(image_paths)} images "
          f"(enhancement + conf >= {conf_threshold})...")

    for img_path in image_paths:
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        enhanced_rgb = clahe_unsharp_enhance(image_rgb)

        enhanced_path = os.path.join(enhanced_dir, os.path.basename(img_path).rsplit(".", 1)[0] + "_enhanced.png")
        cv2.imwrite(enhanced_path, cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR))

        results = model.predict(enhanced_path, conf=conf_threshold, imgsz=imgsz, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf.item())
                coco_cls = model.names[int(box.cls.item())]
                if coco_cls in class_map:
                    project_cls = class_map[coco_cls]
                    all_detections.append((conf, project_cls, img_path, enhanced_path, coco_cls))
                    class_confidences[project_cls].append(conf)

    return all_detections, class_confidences


def summarize_class_confidences(class_confidences: dict) -> pd.DataFrame:
    rows = []
    for cls in sorted(class_confidences.keys()):
        confs = class_confidences[cls]
        rows.append({
            "class": cls,
            "n_detections": len(confs),
            "mean_confidence": float(np.mean(confs)) if confs else 0.0,
            "std_confidence": float(np.std(confs)) if confs else 0.0,
        })
    df = pd.DataFrame(rows).sort_values("mean_confidence", ascending=False).reset_index(drop=True)
    return df


def visualize_top_k(
    model: YOLO,
    detections: list[tuple],
    class_map: dict,
    conf_threshold: float,
    top_k: int,
    output_path: str,
) -> pd.DataFrame:
    top_dets = sorted(detections, key=lambda d: d[0], reverse=True)[:top_k]
    if not top_dets:
        print("No detections above the confidence threshold; skipping visualization.")
        return pd.DataFrame(columns=["confidence", "class", "coco_class", "image_path"])

    n_cols = min(5, len(top_dets))
    n_rows = int(np.ceil(len(top_dets) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 4.4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, (conf, project_cls, img_path, enhanced_path, coco_cls) in enumerate(top_dets):
        image = cv2.cvtColor(cv2.imread(enhanced_path), cv2.COLOR_BGR2RGB)
        results_vis = model.predict(enhanced_path, conf=conf_threshold, verbose=False)
        color = BOX_COLORS[i % len(BOX_COLORS)]

        for r in results_vis:
            if r.boxes is None:
                continue
            for box in r.boxes:
                box_coco = model.names[int(box.cls.item())]
                if box_coco in class_map:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                    label = f"{class_map[box_coco]} {box.conf.item():.2f}"
                    cv2.putText(image, label, (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        axes[i].imshow(image)
        axes[i].set_title(f"{project_cls} ({coco_cls})\nConf: {conf:.3f}", fontsize=11)
        axes[i].axis("off")

    for j in range(len(top_dets), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return pd.DataFrame(
        [(c, pc, cc, ip) for c, pc, ip, _, cc in top_dets],
        columns=["confidence", "class", "coco_class", "image_path"],
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = YOLO(args.weights)
    print(f"Loaded model: {args.weights}")

    class_map = load_class_map(args.class_map_json)
    image_paths = collect_images(args.images_dir, args.fallback_images_dir, args.fallback_limit)
    if not image_paths:
        raise FileNotFoundError("No images found to evaluate (check --images_dir / --fallback_images_dir).")

    detections, class_confidences = run_confidence_analysis(
        model, image_paths, class_map, args.conf_threshold, args.imgsz, args.output_dir,
    )

    summary_df = summarize_class_confidences(class_confidences)
    print("\nMean detection confidence per class:")
    print(summary_df.to_string(index=False))

    summary_csv = os.path.join(args.output_dir, "per_class_confidence.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved per-class confidence summary to {summary_csv}")

    fig_path = os.path.join(args.output_dir, f"top_{args.top_k}_detections.png")
    top_df = visualize_top_k(model, detections, class_map, args.conf_threshold, args.top_k, fig_path)
    print(f"Saved top-{args.top_k} detection grid figure to {fig_path}")

    top_csv = os.path.join(args.output_dir, f"top_{args.top_k}_detections.csv")
    top_df.to_csv(top_csv, index=False)
    print(f"Saved top-{args.top_k} detection table to {top_csv}")


if __name__ == "__main__":
    main()
