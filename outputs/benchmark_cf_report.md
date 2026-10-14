# FINAL CF Benchmark Report — Held-Out Test Split

## Setup
- Backbone: DINOv2-Small (ViT-S/14), frozen
- Head: `models/dinov2_authenticity_head_cf.pth`
- Calibration: `models/calibration_params_cf.json` (T = 0.7867, pre-fitted)
- OOD: `models/ood_params_cf.json` — NOT AVAILABLE
- Test split: `dataset/cf_small/test` — 400 images (200 authentic + 200 ai_edited)

## 1. Classification (Test, N=400)

| Metric | Value |
|--------|-------|
| Accuracy | 98.50% |
| Macro Precision | 98.52% |
| Macro Recall | 98.50% |
| Macro F1 | 98.50% |
| ROC-AUC | 99.94% |

Confusion Matrix:
```
[[199   1]
 [  5 195]]
```

Per-class:
- **ai_edited**: P=97.55%, R=99.50%, F1=98.51%, support=200
- **authentic**: P=99.49%, R=97.50%, F1=98.48%, support=200

## 2. Calibration (Test, pre-fitted T=0.7867)

| Metric | Value |
|--------|-------|
| ECE | 4.73% |
| NLL | 0.0673 |
| Brier | 0.0279 |
| Mean Entropy | 0.2292 bits |

## 3. OOD

{
  "status": "SKIPPED \u2014 ood_params_cf.json not found"
}

## 4. TTA (Horizontal Flip)

| Metric | Value |
|--------|-------|
| Single-crop accuracy | 98.50% |
| TTA accuracy | 98.50% |
| Agreement rate | 99.50% |
| Disagreements | 2 |

## Validation vs Test Distinction

- **Validation (N=200)**: used for checkpoint selection and calibration fitting only.
- **Test (N=400)**: strictly held out; first and only evaluation in this report.
- No test data was used for training, calibration, OOD thresholding, or model selection.

## Artifacts
- `outputs\benchmark_cf\benchmark_cf_metrics.json`
- `outputs\benchmark_cf\confusion_matrix_cf.png`
- `outputs\benchmark_cf_report.md`
