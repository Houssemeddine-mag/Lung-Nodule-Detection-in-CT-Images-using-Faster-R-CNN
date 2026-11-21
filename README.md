# Lung Nodule Detection in CT Images

A cross-validated object-detection study that localizes pulmonary nodules on 2D CT slices. The primary model is an ensemble of five **Faster R-CNN v2 (ResNet-50 FPN)** detectors; **YOLOv8** is trained under the same folds as a comparative baseline. Predictions from multiple folds are combined with Weighted Boxes Fusion, and the trained ensemble is exposed through a small Streamlit inference demo.

> **Research prototype only.** This repository is not a medical device and must not be used for clinical diagnosis.

## Problem and experimental design

- **Task:** predict axis-aligned bounding boxes around nodules in individual CT images.
- **Labels:** YOLO-format annotations, collapsed to a single `nodule` class for the main experiments.
- **Data split:** 102 images for training/validation and 18 strictly held-out test images.
- **Validation:** 5-fold stratified cross-validation on the 102-image pool; the test set is evaluated once, after model and configuration selection.
- **Reproducibility:** fixed seed (`42`) across folds, fixed training schedules, and out-of-fold predictions used to select detection thresholds.

The full experimental pipeline — data preparation, training, cross-validation, and final evaluation — is in [`lung-nodule-detection-in-ct-images.ipynb`](lung-nodule-detection-in-ct-images.ipynb).

## Models

### Faster R-CNN v2 (primary)

- `fasterrcnn_resnet50_fpn_v2` initialized from COCO weights, with a single-class box predictor and a custom 5-scale anchor grid (sizes 16–256, aspect ratios 0.5/1.0/2.0).
- Input resized to 800 px; up to 50 detections per image with a low pre-NMS score threshold (`0.001`) so that candidate boxes remain available for fusion.
- Training: 25 epochs, SGD with momentum `0.9`, learning rate `0.004`, weight decay `1e-3`, 2-iteration linear warmup followed by cosine decay, and mixed-precision training.
- Regularization: horizontal/vertical flips, box-safe crop-zoom, brightness and contrast jitter, plus exponential moving average (EMA, decay `0.98`) of model weights. The final EMA weights of each fold are saved as `app/models/frcnn_f{0..4}.pt`.

### YOLOv8 (baseline)

- Four configurations were compared by out-of-fold mAP50: `s640`, `s1024`, `m1024`, and `s1024_aug` (extra scale, rotation, and HSV jitter).
- Training: 60 epochs with early stopping (patience 20), batch size 8, weight decay `1e-3`, and vertical flip augmentation.
- `s640` achieved the best out-of-fold mAP50 and was selected for the held-out comparison; its five fold weights are stored under `yolo_baseline/`.

### Inference and fusion

- Each fold model produces raw detections; optional horizontal-flip test-time augmentation (TTA) doubles the prediction set.
- Predictions are merged with **Weighted Boxes Fusion** (`IoU = 0.5`, `skip_box_thr = 0.001`) rather than simple NMS.
- Decision thresholds are tuned on out-of-fold F1 scores and recorded in `app/models/ensemble_config.json`. The Streamlit demo exposes the confidence threshold as an interactive control.

## Held-out test results

Metrics are COCO-style `mAP50` / `mAP` (IoU 0.50 and 0.50:0.95), with precision, recall, and F1 at the selected operating point. Full details are in [`results/final_test_results.csv`](results/final_test_results.csv).

| Method | mAP50 | mAP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Faster R-CNN, single model (mean of 5 folds) | 0.577 | 0.269 | 0.682 | 0.558 | **0.610** |
| Faster R-CNN, 5-fold ensemble | **0.679** | **0.322** | 0.875 | 0.368 | 0.519 |
| Faster R-CNN, 5-fold ensemble + TTA | 0.609 | 0.284 | 0.800 | 0.421 | 0.552 |
| YOLOv8, single model (mean of 5 folds) | 0.508 | 0.242 | 0.684 | 0.516 | 0.582 |
| YOLOv8, 5-fold ensemble | 0.588 | 0.271 | 0.750 | 0.474 | 0.581 |
| Faster R-CNN + YOLOv8, 10-model ensemble + TTA | 0.628 | 0.316 | 0.778 | 0.368 | 0.500 |

The Faster R-CNN fold ensemble gives the strongest localization quality (mAP50 0.679) and the highest precision (0.875), while single-fold models achieve the best F1. Performance varies with the precision/recall trade-off, which is why the deployment threshold is configurable.

### YOLO configuration sweep (out-of-fold)

| Configuration | OOF mAP50 | Fold mAP50 mean ± std |
|---|---:|---:|
| **s640 (selected)** | **0.490** | 0.515 ± 0.124 |
| s1024 | 0.468 | 0.490 ± 0.107 |
| m1024 | 0.377 | 0.391 ± 0.100 |
| s1024_aug | 0.425 | 0.462 ± 0.106 |

## Repository layout

```text
data/                       Source dataset archive
preprocessing/              Dataset exploration notebook and archive
lung-nodule-detection-in-ct-images.ipynb   Full training/evaluation pipeline
app/                        Streamlit demo, detector wrapper, and Faster R-CNN fold weights
yolo_baseline/              YOLOv8 fold weights and training log for the selected configuration
results/                    Final test metrics and YOLO sweep summary
```

Model weights (`.pt`) are stored with **Git LFS**; install Git LFS before cloning to retrieve them.

## Running the demo

```powershell
cd app
pip install -r requirements.txt
streamlit run app.py
```

Upload a CT slice to visualize predicted boxes and confidence scores. An optional YOLO-format label file overlays ground-truth boxes in green for side-by-side inspection.

## Scope and limitations

- Detection operates on 2D slices, not full CT volumes; no 3D temporal or cross-slice consistency is applied.
- Experiments use a small dataset (120 images total), so metrics have notable fold-to-fold variance.
- The single-class setup does not distinguish nodule subtypes; the notebook supports a two-class variant.
- The demo is intended for method demonstration and qualitative inspection only.
