"""
generate_pseudo_labels.py
==========================
Stage 1 of pseudo-labeling: run a pretrained "teacher" detector (YOLOv8s,
COCO-pretrained) over the CycleGAN-enhanced synthetic images and store its
raw predictions as *candidate* labels, including per-box confidence.

Candidate labels are written in an extended YOLO format
    <class_id> <x_center> <y_center> <width> <height> <confidence>
(coordinates normalized to [0, 1]) so that the subsequent filtering stage
(`label_filtering.py`) can apply and sweep confidence thresholds without
re-running inference. A permissive default threshold (0.25, matching the
original experimental notebook) is used here on purpose: all selection
decisions are deferred to the filtering stage.

Class-space handling
---------------------
A COCO-pretrained teacher predicts COCO class ids (0=person, 1=bicycle,
2=car, ...). Writing those ids directly into YOLO labels would silently
mix the COCO label space with the project's own class space (the original
notebook did exactly this). By default this script therefore REMAPS COCO
class names to the project taxonomy via `--class_map_json` / DEFAULT_CLASS_MAP
and discards predictions of unmapped classes. Pass `--keep_raw_coco_ids`
to reproduce the original (un-remapped) behaviour for comparison.

Usage:
    python generate_pseudo_labels.py \
        --images_dir /content/synthetic_enhanced \
        --output_dir /content/pseudo_labels_raw \
        --teacher_weights yolov8s.pt --conf 0.25
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

from ultralytics import YOLO

# Project taxonomy order (see ../dataset_preparation/config.py).
TAXONOMY_CLASSES = [
    "Bicycle", "Bridge", "Bus", "Car", "Chimney", "Crosswalk",
    "Hydrant", "Motorcycle", "Other", "Palm", "Stair", "Traffic Light",
]
TAXONOMY_TO_ID = {c: i for i, c in enumerate(TAXONOMY_CLASSES)}

# COCO class name -> project class name. Only classes with a direct
# counterpart in the reCAPTCHAv2 taxonomy are retained.
DEFAULT_CLASS_MAP = {
    "bicycle": "Bicycle",
    "car": "Car",
    "motorcycle": "Motorcycle",
    "bus": "Bus",
    "traffic light": "Traffic Light",
    "fire hydrant": "Hydrant",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate candidate pseudo-labels with a teacher detector.")
    p.add_argument("--images_dir", type=str, required=True,
                    help="Directory of CycleGAN-enhanced images (output of cyclegan/generate_synthetic.py).")
    p.add_argument("--output_dir", type=str, default="./pseudo_labels_raw")
    p.add_argument("--teacher_weights", type=str, default="yolov8s.pt")
    p.add_argument("--conf", type=float, default=0.25,
                    help="Permissive inference threshold; final selection happens in label_filtering.py.")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold used by the teacher.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--class_map_json", type=str, default=None,
                    help="JSON {coco_name: project_class_name} overriding DEFAULT_CLASS_MAP.")
    p.add_argument("--keep_raw_coco_ids", action="store_true",
                    help="Write raw COCO class ids (reproduces the original notebook behaviour).")
    p.add_argument("--stats_json", type=str, default="generation_stats.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    class_map = DEFAULT_CLASS_MAP
    if args.class_map_json:
        with open(args.class_map_json, "r") as f:
            class_map = json.load(f)

    image_paths = sorted(
        glob.glob(os.path.join(args.images_dir, "*.png"))
        + glob.glob(os.path.join(args.images_dir, "*.jpg"))
        + glob.glob(os.path.join(args.images_dir, "*.jpeg"))
    )
    if not image_paths:
        sys.exit(f"No images found in {args.images_dir}")

    teacher = YOLO(args.teacher_weights)
    print(f"Teacher: {args.teacher_weights} | images: {len(image_paths)} | conf >= {args.conf}")

    n_raw_boxes, n_kept_boxes, n_images_with_boxes = 0, 0, 0
    per_class = Counter()

    for img_path in image_paths:
        result = teacher.predict(img_path, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
        lines = []
        if result.boxes is not None:
            for box in result.boxes:
                n_raw_boxes += 1
                coco_id = int(box.cls[0].item())
                coco_name = teacher.names[coco_id]
                conf = float(box.conf[0].item())
                xc, yc, w, h = box.xywhn[0].cpu().numpy()

                if args.keep_raw_coco_ids:
                    cls_id, cls_name = coco_id, coco_name
                else:
                    if coco_name not in class_map:
                        continue
                    cls_name = class_map[coco_name]
                    cls_id = TAXONOMY_TO_ID[cls_name]

                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.4f}")
                per_class[cls_name] += 1
                n_kept_boxes += 1

        if lines:
            n_images_with_boxes += 1
        out_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        with open(out_path, "w") as f:
            f.write("\n".join(lines))

    stats = {
        "teacher": args.teacher_weights,
        "conf_threshold": args.conf,
        "num_images": len(image_paths),
        "num_images_with_candidate_boxes": n_images_with_boxes,
        "raw_teacher_boxes": n_raw_boxes,
        "candidate_boxes_after_class_mapping": n_kept_boxes,
        "class_remapped": not args.keep_raw_coco_ids,
        "boxes_per_class": dict(per_class),
    }
    with open(os.path.join(args.output_dir, args.stats_json), "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"\nCandidate labels written to {args.output_dir}")
    print("Next step: python label_filtering.py --raw_labels_dir " + args.output_dir)


if __name__ == "__main__":
    main()
