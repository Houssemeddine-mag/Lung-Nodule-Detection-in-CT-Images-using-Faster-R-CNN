# Lung Nodule Detection in CT Images

A comparative object-detection study that localizes pulmonary nodules on 2D CT slices. The project trains several detector families under one protocol — two Faster R-CNN variants, two from-scratch CNN baselines, a YOLOv8 baseline, and an RF-DETR baseline — and measures them on the same held-out test set. The strongest configuration is a cross-architecture consensus ensemble of Faster R-CNN and RF-DETR, exposed through a Streamlit demo.

> **Research prototype only.** This repository is not a medical device and must not be used for clinical diagnosis.

## Experimental protocol

- **Task:** axis-aligned bounding-box detection of nodules in individual CT images, collapsed to a single `nodule` class.
- **Data:** 102 images for training/validation and 18 strictly held-out test images.
- **Validation:** 5-fold stratified cross-validation on the 102-image pool; the test set is evaluated once, after configuration selection.
- **Reproducibility:** fixed seed (`42`), fixed schedules, and detection thresholds tuned on out-of-fold predictions only.
- **Metrics:** COCO-style `mAP50` (IoU 0.50) and `mAP` (IoU 0.50:0.95), plus precision, recall, and F1 at the chosen operating point. Some experiments also report false positives per image (`FP/img`).

Notebooks:

- [`lung-nodule-detection-in-ct-images.ipynb`](lung-nodule-detection-in-ct-images.ipynb) — original Faster R-CNN + YOLOv8 study.
- [`Baseline/baseline_cnn_models 01/baseline-cnn-lung-nodule-detection-structured.ipynb`](Baseline/baseline_cnn_models%2001/baseline-cnn-lung-nodule-detection-structured.ipynb) — first plain-CNN baseline.
- [`Baseline/baseline_cnn_models 02/baseline-cnn-lung-nodule%202.0.ipynb`](Baseline/baseline_cnn_models%2002/baseline-cnn-lung-nodule%202.0.ipynb) — refined plain-CNN baseline.
- [`preprocessing/lungdataexploration.ipynb`](preprocessing/lungdataexploration.ipynb) — dataset exploration.

The RF-DETR / Tiny-Nodule experiments are distributed as exported weights and result CSVs under `app 2/deliver_reliable/`; their training notebook is not included in this repository.

---

## Models

### 1. Faster R-CNN v2 — original ResNet-50 FPN detector

**Where:** `app/first/` (weights and first demo), results in `app/results/`.

The first detector uses `fasterrcnn_resnet50_fpn_v2` initialized from COCO weights, a single-class box predictor, and a custom anchor grid (sizes 16–256 px, aspect ratios 0.5/1.0/2.0). Inputs are resized to 800 px; up to 50 low-threshold candidates are kept for fusion.

Training uses 25 epochs, SGD (lr `0.004`, momentum `0.9`, weight decay `1e-3`), a 2-iteration warmup followed by cosine decay, mixed precision, and EMA weights (decay `0.98`). Augmentation: horizontal/vertical flips, box-safe crop-zoom, brightness and contrast jitter. Inference combines fold predictions with Weighted Boxes Fusion (IoU `0.5`, `skip_box_thr = 0.001`) and optional horizontal-flip TTA.

Held-out test results ([`app/results/final_test_results.csv`](app/results/final_test_results.csv)):

| Variant | mAP50 | mAP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Single model (mean of 5 folds, std mAP50 0.035) | 0.577 | 0.269 | 0.682 | 0.558 | **0.610** |
| 5-fold ensemble | **0.679** | **0.322** | **0.875** | 0.368 | 0.519 |
| 5-fold ensemble + flip TTA | 0.609 | 0.284 | 0.800 | 0.421 | 0.552 |

The fold ensemble gives the best localization accuracy and highest precision; single-fold models reach the best F1. The decision threshold is stored in `app/first/models/ensemble_config.json` (`0.65`).

### 2. Faster R-CNN v2 — Tiny-Nodule specialization

**Where:** `app 2/deliver_reliable/models/frcnn/` (weights), results in `models/results/`.

A second Faster R-CNN v2 is tuned for small nodules. It runs at 832 px and replaces the default anchors with five finer scales — `(12,16,24)`, `(24,32,48)`, `(48,64,96)`, `(96,128,160)`, `(192,256,320)` — and aspect ratios `0.7/1.0/1.4`. The RPN head uses `conv_depth = 2`; the box head keeps 30 detections per image with `score_thresh = 0.01` and NMS IoU `0.45`. Five fold models are fused with WBF and four-way TTA (none, horizontal, vertical, both).

Out-of-fold results ([`cv_comparison_oof.csv`](app%202/deliver_reliable/models/results/cv_comparison_oof.csv)):

| Variant | OOF mAP50 | OOF mAP | Fold mAP50 mean ± std | Thr (F1) | P | R | F1 | FP/img |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tiny-Nodule | 0.586 | 0.256 | 0.597 ± 0.115 | 0.35 | 0.698 | 0.571 | 0.628 | 0.255 |
| Tiny-Nodule + 4-way TTA | 0.664 | 0.285 | 0.668 ± 0.127 | 0.375 | 0.825 | 0.629 | **0.714** | 0.137 |

A separate low-false-positive operating point is available (threshold `0.9`: precision `0.825`, recall `0.495`, `0.108` FP/img; with TTA, threshold `0.65`: precision `0.900`, recall `0.514`, `0.059` FP/img).

Held-out test results ([`final_test_results.csv`](app%202/deliver_reliable/models/results/final_test_results.csv)):

| Variant | mAP50 | mAP | P | R | F1 | TP / FP / FN | FP/img |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single model (mean of 5, std mAP50 0.065) | 0.530 | 0.250 | 0.661 | 0.579 | 0.617 | 11.0 / 5.6 / 8.0 | 0.311 |
| 5-fold ensemble + 4-way TTA | 0.599 | 0.295 | 0.733 | 0.579 | 0.647 | 11.0 / 4.0 / 8.0 | 0.222 |

### 3. Baseline CNN v1 — plain convolutional detector from scratch

**Where:** [`Baseline/baseline_cnn_models 01/`](Baseline/baseline_cnn_models%2001/).

This baseline deliberately avoids pretrained weights and ready-made detectors. A 416×416 image passes through four convolution blocks (Conv → BatchNorm → ReLU → MaxPool), one extra convolution, and a 1×1 head producing a 26×26 grid (stride 16). Each cell predicts objectness plus box offsets: about 0.25 M parameters in total.

Loss = class-weighted BCE over all 676 grid cells (positive weight `20`) plus L1 box loss weighted by `5`. Augmentation is limited to flips and brightness changes. Training uses AdamW (lr `1e-3`, weight decay `1e-3`), 60 epochs, batch size 8, cosine decay, and NMS IoU `0.5`.

Results ([`baseline_cnn_results.csv`](Baseline/baseline_cnn_models%2001/models/baseline_cnn_results.csv)):

| Evaluation | mAP50 | mAP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| CV, pooled out-of-fold | 0.136 | 0.042 | 0.241 | 0.200 | 0.219 |
| CV folds, mean ± std mAP50 | 0.174 ± 0.090 | 0.062 | 0.253 | 0.206 | 0.190 |
| Test, mean over 5 fold models | 0.174 | 0.073 | 0.242 | 0.200 | 0.200 |

This establishes the lower bound: the simple grid detector finds some nodules but frequently produces weak or duplicated boxes.

### 4. Baseline CNN v2 — refined plain convolutional detector

**Where:** [`Baseline/baseline_cnn_models 02/`](Baseline/baseline_cnn_models%2002/).

The second CNN keeps the same architecture family but changes the training setup: 120 epochs, batch size 4, AdamW lr `5e-4` with weight decay `5e-3`, and cosine decay. The grid is denser — stride 8, giving 52×52 cells — with the final conv block left unpooled. Box scale is reduced to `0.10 × IMG`, the positive-objectness weight is raised to `150`, box-loss weight to `10`, and NMS IoU lowered to `0.3` to suppress duplicates.

Results ([`baseline_cnn_results.csv`](Baseline/baseline_cnn_models%2002/models/baseline_cnn_results.csv)):

| Evaluation | mAP50 | mAP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| CV, pooled out-of-fold (threshold 0.92) | **0.241** | **0.095** | 0.302 | 0.362 | 0.329 |
| CV folds, mean ± std mAP50 | 0.279 ± 0.067 | 0.117 | 0.322 | 0.364 | 0.334 |
| Test, mean over 5 fold models | **0.357** | **0.178** | 0.330 | 0.484 | 0.386 |

Per-fold test mAP50 values are `0.369, 0.245, 0.448, 0.305, 0.421`. The refinements roughly double test mAP50 over v1, but the model still trails the two-stage and transformer detectors by a wide margin. Note that the exported `baseline_detector.py` decodes with the v1 26×26/stride-16 grid; the v2 notebook trains with a 52×52/stride-8 grid.

### 5. YOLOv8 — one-stage baseline

**Where:** `app/yolo_baseline/`, results in `app/results/`.

Four configurations were compared by out-of-fold mAP50: `s640`, `s1024`, `m1024`, and `s1024_aug`. Training used 60 epochs with early stopping (patience 20), batch size 8, weight decay `1e-3`, and vertical flips. `s640` was selected (OOF mAP50 `0.490`; see [`yolo_sweep_cv.csv`](app/results/yolo_sweep_cv.csv)). On the held-out set, the 5-fold ensemble reaches mAP50 `0.588` / mAP `0.271` / F1 `0.581`.

### 6. RF-DETR Medium and the final consensus ensemble

**Where:** `app 2/deliver_reliable/models/rfdetr/`.

RF-DETR Medium (DINOv2 ViT backbone, resolution 576) supplies the second architecture for the final deliverable. Predictions from 5 RF-DETR folds and 5 Faster R-CNN Tiny folds are fused using weighted WBF (RF-DETR weight `1.5`, Faster R-CNN weight `1.0`, IoU `0.45`) with four-way TTA.

| Variant (OOF) | mAP50 | mAP | F1 (F1-opt threshold) |
|---|---:|---:|---:|
| RF-DETR Medium | 0.617 | 0.296 | 0.659 |
| RF-DETR Medium + 4-way TTA | 0.630 | 0.285 | 0.684 |
| Consensus ensemble + TTA | **0.690** | 0.292 | **0.731** |

| Variant (held-out test) | mAP50 | mAP | P | R | F1 | FP/img |
|---|---:|---:|---:|---:|---:|---:|
| RF-DETR single (mean of 5, std 0.048) | 0.658 | 0.318 | 0.794 | 0.568 | 0.660 | 0.156 |
| RF-DETR ensemble + 4-way TTA | 0.719 | 0.382 | 0.765 | 0.684 | 0.722 | 0.222 |
| Consensus ×10 + TTA (F1-opt, thr 0.15) | **0.726** | 0.366 | 0.750 | 0.789 | **0.769** | 0.278 |
| Consensus ×10 + TTA (low-FP, thr 0.25) | **0.726** | 0.366 | **0.812** | 0.684 | 0.743 | **0.167** |

## Progression on the held-out test set

| Model | Test mAP50 | Test F1 |
|---|---:|---:|
| Baseline CNN v1 (mean fold model) | 0.174 | 0.200 |
| Baseline CNN v2 (mean fold model) | 0.357 | 0.386 |
| YOLOv8 s640 (5-fold ensemble) | 0.588 | 0.581 |
| Faster R-CNN v2 Tiny + TTA (5-fold ensemble) | 0.599 | 0.647 |
| Faster R-CNN v2 original (5-fold ensemble) | 0.679 | 0.519 |
| RF-DETR Medium + TTA (5-fold ensemble) | 0.719 | 0.722 |
| Faster R-CNN + RF-DETR consensus + TTA | **0.726** | **0.769** |

The two plain CNNs validate that the task is non-trivial; pretrained two-stage and transformer detectors improve localization substantially, and cross-architecture fusion gives the best recall and F1.

## Repository layout

```text
Baseline/                            Two from-scratch CNN baselines (code, notebooks, weights, results)
  baseline_cnn_models 01/            First CNN: 26x26 grid, 60 epochs
  baseline_cnn_models 02/            Refined CNN: 52x52 grid, 120 epochs
app/first/                           Original Faster R-CNN v2 app and fold weights
app/results/                         Original study test metrics and YOLO sweep
app/yolo_baseline/                   Selected YOLOv8 s640 fold weights
app 2/deliver_reliable/              Final deliverable: Tiny-Nodule FRCNN + RF-DETR consensus app
  models/frcnn/                      5 Faster R-CNN v2 Tiny-Nodule weights
  models/rfdetr/                     5 RF-DETR Medium weights
  models/results/                    OOF comparison, test metrics, operating points
Full data/                           Source dataset archive
preprocessing/                       Dataset exploration notebook and archive
CT images/                           Sample CT slices for the demos
lung-nodule-detection-in-ct-images.ipynb   Original FRCNN + YOLO study
```

All `.pt` and `.pth` weights are tracked with **Git LFS**; install Git LFS before cloning to retrieve them.

## Running the demos

Two Streamlit apps are provided:

```powershell
# Original Faster R-CNN v2 ensemble
cd "app/first"
pip install -r requirements.txt
streamlit run app.py

# Final Tiny-Nodule FRCNN + RF-DETR consensus ensemble
cd "app 2/deliver_reliable"
pip install -r requirements.txt
streamlit run app.py
```

Both apps accept a CT slice upload and draw predicted boxes with confidence scores. An optional YOLO-format label file overlays ground truth in green. The final app offers three inference modes (full consensus, consensus without TTA, RF-DETR only) and two operating points (balanced F1 or low false positives).

## Scope and limitations

- Detection operates on 2D slices, not full CT volumes; no cross-slice consistency is applied.
- The dataset is small (120 images total), so fold-to-fold variance is substantial.
- All main experiments use one class; nodule subtypes are not distinguished.
- The CNN baseline v2 notebook and its exported detector use different grid resolutions (see above).
- These models are research artifacts for methodological comparison only.
