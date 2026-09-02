# Evaluation Module

This directory implements the quantitative and qualitative evaluation of
the trained YOLOv8 detector (`../yolov8/`) produced by the *Breaking
reCAPTCHAv2* pipeline. It reports standard object-detection metrics
(mAP@0.5, mAP@0.5:0.95, precision, recall, per-class AP), an
IoU-matched confusion matrix, and a test-time-enhancement-based confidence
analysis with qualitative detection visualizations — the combination of
evidence typically expected in the Results section of a Q1-level
computer-vision paper.

## 1. Evaluation Protocol Overview

```mermaid
flowchart TD
    W[Trained weights\nbest.pt / last.pt] --> M1[compute_metrics.py\nmAP@0.5, mAP@0.5:0.95,\nprecision, recall, per-class AP]
    W --> M2[confusion_matrix.py\nIoU-matched multi-class\nconfusion matrix]
    W --> M3[confidence_analysis.py\nCLAHE+unsharp test-time enhancement,\nper-class mean confidence,\ntop-K qualitative detections]
    M1 --> R[results/\nmetrics_summary.json, per_class_ap.csv]
    M2 --> R
    M3 --> R
```

Each script is independent and can be run against any checkpoint produced
by `../yolov8/train_yolov8.py`; together they provide three complementary
views of model performance:

| Script | Question answered | Output |
|---|---|---|
| `compute_metrics.py` | How accurate is the detector, in standard COCO-style terms? | `metrics_summary.json`, `per_class_ap.csv`, Ultralytics PR-curve/confusion-matrix plots |
| `confusion_matrix.py` | Which classes does the detector confuse with which, and how many detections are false positives/negatives? | `confusion_matrix.png`, `confusion_matrix.csv` |
| `confidence_analysis.py` | Under test-time enhancement, which classes does the model detect most confidently, and what do its best detections look like? | `per_class_confidence.csv`, `top_K_detections.png`, `top_K_detections.csv` |

## 2. Quantitative Metrics (`compute_metrics.py`)

Computed via Ultralytics' native `YOLO.val()` routine on the held-out
validation split defined in `dataset.yaml`:

- **mAP@0.5** — mean Average Precision at a single IoU threshold of 0.5
  (PASCAL-VOC-style criterion).
- **mAP@0.5:0.95** — mean Average Precision averaged over IoU thresholds
  0.5 to 0.95 in steps of 0.05 (the primary COCO detection metric),
  penalizing loosely localized boxes more heavily than mAP@0.5.
- **Precision / Recall** — box-level precision and recall, macro-averaged
  over classes.
- **Per-class AP@0.5 and AP@0.5:0.95** — exported to `per_class_ap.csv` for
  direct inclusion as a results table in the paper.

### Handling of unannotated validation images

As documented in `../dataset_preparation/README.md`, a subset of this
dataset's images carry only folder-inferred category labels and have
*empty* ground-truth `.txt` files. When such images dominate or fully
compose the validation split, precision/recall/mAP are not mathematically
well-defined (there is no ground truth to score against). Rather than
silently reporting a misleading `0.0000`, `compute_metrics.py` explicitly
detects this case and reports `N/A`, together with a console/JSON note —
this distinction should be preserved when the numbers are transcribed into
the manuscript's results table, to avoid an incorrect reading of "the
model achieves 0% precision."

## 3. IoU-Matched Confusion Matrix (`confusion_matrix.py`)

Rather than relying solely on Ultralytics' internal (less-documented)
confusion-matrix computation, this script implements and documents its own
matching procedure explicitly, so that it is fully reproducible and citable:

1. Predicted boxes are sorted by descending confidence.
2. Each prediction is greedily matched to the highest-IoU unmatched
   ground-truth box of *any* class, provided IoU ≥ `--iou_thr` (default
   `0.5`).
3. A matched (ground-truth class, predicted class) pair populates the
   corresponding confusion-matrix cell (on-diagonal = correct
   classification; off-diagonal = misclassification).
4. An unmatched prediction is a **false positive**, recorded in the
   `background` column.
5. An unmatched ground-truth box is a **false negative**, recorded in the
   `background` row.

This yields an (N+1) × (N+1) matrix (N project classes + one explicit
`background` category), rendered as a normalized heatmap
(`confusion_matrix.png`) and exported as a raw-count CSV
(`confusion_matrix.csv`).

## 4. Test-Time-Enhancement Confidence Analysis (`confidence_analysis.py`)

This analysis is deliberately distinct from the CycleGAN-based enhancement
used at *training* time (`../cyclegan/`): it applies a lightweight,
non-learned transform — CLAHE contrast normalization on the LAB color
space's L channel, followed by unsharp masking — to each validation image
**at inference time only**, before running detection at a high confidence
threshold (default `0.80`). This isolates test-time image conditioning as
an independent factor from the detector's learned representations, a
standard control in image-quality/detection ablation studies.

Because the underlying detector backbone (YOLOv8s, COCO-pretrained
initialization) predicts COCO class names, detections are remapped onto
the project's reCAPTCHAv2 taxonomy via an explicit class map:

```python
DEFAULT_CLASS_MAP = {
    "person": "Other", "bicycle": "Bicycle", "car": "Car",
    "motorcycle": "Motorcycle", "bus": "Bus",
    "traffic light": "Traffic Light", "fire hydrant": "Hydrant",
}
```

(Override with `--class_map_json` for a custom mapping.)

For each project class, the mean and standard deviation of detection
confidence across all qualifying detections are reported
(`per_class_confidence.csv`), and the top-K (default 10) highest-confidence
detections across the whole validation set are rendered as an annotated
image grid (`top_K_detections.png`) — directly usable as the paper's
qualitative results figure.

## 5. Repository Contents

```
evaluation/
├── README.md                 # this file
├── enhancement.py             # shared CLAHE + unsharp-mask enhancement utility
├── compute_metrics.py         # mAP@0.5, mAP@0.5:0.95, precision, recall, per-class AP
├── confusion_matrix.py        # IoU-matched multi-class confusion matrix
└── confidence_analysis.py     # test-time-enhancement confidence analysis + qualitative figure
```

## 6. Usage

### 6.1 Environment

```bash
pip install ultralytics opencv-python matplotlib numpy pandas
```

### 6.2 Running the full evaluation suite

```bash
WEIGHTS=/content/yolo_runs/final_with_cyclegan/weights/best.pt
DATA=/content/yolo_dataset/dataset.yaml
IMAGES=/content/yolo_dataset/images/val
LABELS=/content/yolo_dataset/labels/val

# 1. Standard quantitative metrics
python compute_metrics.py --weights $WEIGHTS --data $DATA --output_dir ./results

# 2. IoU-matched confusion matrix
python confusion_matrix.py --weights $WEIGHTS --images_dir $IMAGES --labels_dir $LABELS \
    --class_names Chimney Crosswalk Stair --output_dir ./results

# 3. Test-time-enhancement confidence analysis + qualitative figure
python confidence_analysis.py --weights $WEIGHTS --images_dir $IMAGES \
    --fallback_images_dir /content/yolo_dataset/images/train --output_dir ./results \
    --conf_threshold 0.80 --top_k 10
```

All three scripts write into a shared `./results/` directory, so the
complete evidence set for a given checkpoint (metrics JSON/CSV, confusion
matrix figure/table, per-class confidence table, qualitative detection
grid) is collected in one place, ready to be pulled directly into the
manuscript's figures and tables.

### 6.3 Output artifacts

| File | Produced by | Description |
|---|---|---|
| `metrics_summary.json` | `compute_metrics.py` | mAP@0.5, mAP@0.5:0.95, macro precision/recall (or `N/A`) |
| `per_class_ap.csv` | `compute_metrics.py` | Per-class AP@0.5 / AP@0.5:0.95 |
| `val/` (Ultralytics plots) | `compute_metrics.py` | Native PR curves, F1 curve, confusion matrix |
| `confusion_matrix.png` / `.csv` | `confusion_matrix.py` | IoU-matched confusion matrix (heatmap + raw counts) |
| `per_class_confidence.csv` | `confidence_analysis.py` | Mean/std detection confidence per class under test-time enhancement |
| `top_K_detections.png` / `.csv` | `confidence_analysis.py` | Annotated qualitative figure of the K highest-confidence detections |

## 7. Reproducibility Notes

- All three scripts take `--weights` explicitly rather than relying on a
  global variable, so results are traceable to a specific checkpoint file.
- `confusion_matrix.py` fully specifies its IoU-matching algorithm in code
  and in this README (Section 3) rather than depending on an external
  library's internal implementation, so the confusion matrix reported in
  the paper is independently reproducible from the description alone.
- `confidence_analysis.py` persists every enhanced image it generates to
  `results/enhanced_images/`, so the exact test-time-enhanced inputs behind
  any reported confidence value can be inspected after the fact.
- The default confidence threshold in `confidence_analysis.py` (`0.80`) is
  intentionally high (isolating the model's most reliable predictions for
  qualitative inspection) and is *not* the same as the default confidence
  threshold used for standard validation in `compute_metrics.py` (Ultralytics'
  internal default, typically swept across a PR curve); this distinction
  should be stated explicitly wherever both numbers appear in the same
  table or figure.
