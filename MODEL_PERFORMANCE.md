# Model Performance — YOLO11n Hand-Gesture Detector

**Architecture:** YOLO11n (nano)  
**Training pipeline:** Two-phase fine-tuning (`train_best.py`) — Phase 1 aggressive augmentation (200 epochs) → Phase 2 clean fine-tune (50 epochs)  
**Export:** INT8 post-training quantisation for Sony IMX500 NPU

| Setting | Value |
|---|---|
| Architecture | YOLO11n (nano) |
| Input resolution | 640 × 640 |
| Classes | 6 (`fist_close`, `index_finger_up`, `open_palm`, `thumbs_down`, `thumbs_up`, `two_fingers`) |
| Dataset | 690 images · 552 train / 138 val (80/20 split) |
| Phase 1 epochs | 200 (AdamW, aggressive augmentation) |
| Phase 2 epochs | 50 (AdamW, clean fine-tune) |
| Optimiser | AdamW |
| Batch size | 16 |

---

## Final validation metrics (Phase 2, epoch 50)

| Metric | Value |
|---|---|
| Precision | **95.9%** |
| Recall | **96.9%** |
| mAP50 | **98.1%** |
| mAP50-95 | **84.4%** |

---

## Training curves

Loss and metric progression across Phase 2 fine-tuning epochs.

<img src="model_performance/results.png" alt="Training results" width="900">

---

## Precision, Recall, F1 and PR curves

<img src="model_performance/BoxPR_curve.png" alt="PR curve" width="900">

<img src="model_performance/BoxP_curve.png" alt="Precision curve" width="900">

<img src="model_performance/BoxR_curve.png" alt="Recall curve" width="900">

<img src="model_performance/BoxF1_curve.png" alt="F1 curve" width="900">

---

## Confusion matrices

Raw counts and row-normalised views across all 6 gesture classes.

<img src="model_performance/confusion_matrix.png" alt="Confusion matrix" width="900">

<img src="model_performance/confusion_matrix_normalized.png" alt="Confusion matrix normalised" width="900">

---

## Dataset label distribution

Class balance across train and validation splits.

<img src="model_performance/labels.jpg" alt="Label distribution" width="900">

---

## Training batch samples

Augmented training samples from Phase 2.

<img src="model_performance/train_batch0.jpg" alt="Train batch 0" width="900">

<img src="model_performance/train_batch1.jpg" alt="Train batch 1" width="900">

<img src="model_performance/train_batch2.jpg" alt="Train batch 2" width="900">

---

## Validation — ground truth vs predictions

**Batch 0 — labels**

<img src="model_performance/val_batch0_labels.jpg" alt="Val batch 0 labels" width="900">

**Batch 0 — predictions**

<img src="model_performance/val_batch0_pred.jpg" alt="Val batch 0 pred" width="900">

**Batch 1 — labels**

<img src="model_performance/val_batch1_labels.jpg" alt="Val batch 1 labels" width="900">

**Batch 1 — predictions**

<img src="model_performance/val_batch1_pred.jpg" alt="Val batch 1 pred" width="900">

**Batch 2 — labels**

<img src="model_performance/val_batch2_labels.jpg" alt="Val batch 2 labels" width="900">

**Batch 2 — predictions**

<img src="model_performance/val_batch2_pred.jpg" alt="Val batch 2 pred" width="900">

---

## Notes

- Images live in the `model_performance/` folder — commit that folder with this file for GitHub.
- If Cursor preview still blocks images: open **`MODEL_PERFORMANCE.html`** in Chrome/Edge (double-click in Explorer).
- INT8 quantisation may lower confidence scores slightly vs FP32 — `pi_tester.py` uses threshold `0.60` vs `0.75` on Windows.
