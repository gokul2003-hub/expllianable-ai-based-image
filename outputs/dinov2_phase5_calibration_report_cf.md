# Phase 5 Calibration Report — Community Forensics-Small

## Methodology
- Backbone: DINOv2-Small (ViT-S/14), frozen
- Head: `models/dinov2_authenticity_head_cf.pth`
- Calibration samples: 200 (validation split ONLY)
- Test split: NOT read during calibration
- Learned temperature: **T = 0.7867**
- Class mapping: `{'ai_edited': 0, 'authentic': 1}`

## Validation Metrics (N=200)

| Metric | Raw Softmax | Temperature Scaled |
| :--- | ---: | ---: |
| ECE | 6.96% | 4.16% |
| NLL | 0.1691 | 0.1541 |
| Brier | 0.0779 | 0.0701 |
| Accuracy | 97.50% | 97.50% |
| Macro F1 | 97.50% | 97.50% |
| ROC-AUC | 98.40% | 98.40% |

Note: Temperature scaling is rank-preserving, so accuracy, F1 and ROC-AUC
are mathematically invariant. Only probability alignment (ECE/NLL/Brier) changes.

## Artifacts
- `models\calibration_params_cf.json`
- `outputs\calibration_cf/reliability_raw.png`
- `outputs\calibration_cf/reliability_calibrated.png`

## Scope Limitation
Metrics above are validation-split metrics. Held-out test evaluation is a
separate step and was deliberately not performed here.
