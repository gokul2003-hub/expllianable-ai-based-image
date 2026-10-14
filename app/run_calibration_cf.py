"""
CF-Specific Phase 5 Calibration & Uncertainty Quantification.

Fits a single scalar temperature T on the Community Forensics-Small
VALIDATION split only (200 images). The test split is never read.

Methodology is identical to app/run_calibration.py:
  - TemperatureScaler (LBFGS on NLL)
  - compute_ece, compute_nll, compute_brier_score, compute_entropy
  - plot_reliability_diagram

Differences from the original runner:
  - Extracts CLS features inline from dataset/cf_small/val (no feature cache)
  - Loads models/dinov2_authenticity_head_cf.pth
  - Writes models/calibration_params_cf.json
  - Never touches dataset/cf_small/test

Usage:
    python -m app.run_calibration_cf
"""

import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from app.models_vfm import (
    load_dinov2_small,
    get_dinov2_transforms,
    DINOv2ClassificationHead,
    EMBEDDING_DIM,
)
from app.calibration import (
    TemperatureScaler,
    compute_ece,
    compute_nll,
    compute_brier_score,
    compute_entropy,
    plot_reliability_diagram,
)


# ─────────────────────────────────────────────────────────────────────────────
# CF-specific configuration
# ─────────────────────────────────────────────────────────────────────────────
VAL_DIR = os.path.join("dataset", "cf_small", "val")
# TEST_DIR is intentionally never referenced.

CKPT_PATH = os.path.join("models", "dinov2_authenticity_head_cf.pth")
CALIB_OUT = os.path.join("models", "calibration_params_cf.json")
PLOT_DIR = os.path.join("outputs", "calibration_cf")
REPORT_PATH = os.path.join("outputs", "dinov2_phase5_calibration_report_cf.md")

EXPECTED_VAL_PER_CLASS = 100
BATCH_SIZE = 32


# ─────────────────────────────────────────────────────────────────────────────
def extract_val_features(device):
    """Extract frozen DINOv2 CLS features from the validation split only."""
    for class_name in ["ai_edited", "authentic"]:
        d = os.path.join(VAL_DIR, class_name)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Missing validation directory: {d}")

    transform = get_dinov2_transforms()
    dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Validation images : {len(dataset)}")
    print(f"  Classes           : {dataset.classes}")
    print(f"  Class mapping     : {dataset.class_to_idx}")

    expected = {"ai_edited": 0, "authentic": 1}
    if dataset.class_to_idx != expected:
        raise ValueError(f"Class mapping mismatch: {dataset.class_to_idx} != {expected}")

    backbone, metadata = load_dinov2_small(device=device)
    print(f"  Backbone          : {metadata['model_name']} (frozen)")

    feats, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            fd = backbone.forward_features(images)
            feats.append(fd["x_norm_clstoken"].cpu())
            labels.append(batch_labels.clone())

    X = torch.cat(feats, dim=0)
    y = torch.cat(labels, dim=0)
    print(f"  Feature matrix    : {tuple(X.shape)}")
    print(f"  Label vector      : {tuple(y.shape)}")

    if X.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Feature dim {X.shape[1]} != {EMBEDDING_DIM}")

    return X, y, dataset.classes


def main():
    print("=" * 70)
    print("  PHASE 5 (CF): CONFIDENCE CALIBRATION — VALIDATION SPLIT ONLY")
    print("=" * 70)

    device = torch.device("cpu")
    os.makedirs("models", exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Validation features
    print("\n[Step 1] Extracting validation features (test split never read)")
    val_feats, val_labels, classes = extract_val_features(device)
    val_feats_np = val_feats.numpy()
    val_labels_np = val_labels.numpy()
    n_samples = len(val_labels_np)
    print(f"  Calibration samples: {n_samples}")

    # 2. Load CF head
    print("\n[Step 2] Loading CF classification head")
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"CF checkpoint not found: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    head = DINOv2ClassificationHead(
        in_features=EMBEDDING_DIM,
        hidden_dim=ckpt["hidden_dim"],
        num_classes=len(classes),
    ).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    print(f"  Head loaded (strict): {CKPT_PATH}")

    # 3. Validation logits
    with torch.no_grad():
        val_logits = head(val_feats.to(device)).cpu().numpy()
    print(f"  Validation logits : {val_logits.shape}")

    # 4. Fit temperature on validation ONLY
    print("\n[Step 3] Fitting TemperatureScaler on validation split")
    scaler = TemperatureScaler()
    best_T = scaler.fit(val_logits, val_labels_np)
    print(f"  --> Learned temperature T = {best_T:.4f}")

    # 5. Save CF calibration params (never overwrite old file)
    with open(CALIB_OUT, "w", encoding="utf-8") as f:
        json.dump(
            {
                "temperature": best_T,
                "model": "DINOv2-Small (ViT-S/14)",
                "dataset": "cf_small",
                "fitted_on": "validation",
                "n_calibration_samples": int(n_samples),
                "classes": classes,
                "class_to_idx": ckpt["class_to_idx"],
            },
            f,
            indent=4,
        )
    print(f"  --> Saved to: {CALIB_OUT}")

    # 6. Pre-calibration validation metrics
    exp_v = np.exp(val_logits - np.max(val_logits, axis=-1, keepdims=True))
    raw_probs = exp_v / np.sum(exp_v, axis=-1, keepdims=True)
    raw_preds = np.argmax(raw_probs, axis=1)
    raw_acc = accuracy_score(val_labels_np, raw_preds)
    raw_prec, raw_rec, raw_f1, _ = precision_recall_fscore_support(
        val_labels_np, raw_preds, average="macro", zero_division=0
    )
    raw_auc = roc_auc_score(val_labels_np, raw_probs[:, 1])
    raw_ece, raw_diag = compute_ece(raw_probs, val_labels_np, n_bins=15)
    raw_nll = compute_nll(raw_probs, val_labels_np)
    raw_brier = compute_brier_score(raw_probs, val_labels_np)
    raw_entropy = compute_entropy(raw_probs)

    # 7. Post-calibration validation metrics
    calib_probs = scaler.predict_proba(val_logits)
    calib_preds = np.argmax(calib_probs, axis=1)
    calib_acc = accuracy_score(val_labels_np, calib_preds)
    calib_prec, calib_rec, calib_f1, _ = precision_recall_fscore_support(
        val_labels_np, calib_preds, average="macro", zero_division=0
    )
    calib_auc = roc_auc_score(val_labels_np, calib_probs[:, 1])
    calib_ece, calib_diag = compute_ece(calib_probs, val_labels_np, n_bins=15)
    calib_nll = compute_nll(calib_probs, val_labels_np)
    calib_brier = compute_brier_score(calib_probs, val_labels_np)
    calib_entropy = compute_entropy(calib_probs)

    # 8. Comparison table
    print("\n" + "=" * 65)
    print(f"CALIBRATION COMPARISON — VALIDATION SPLIT (N={n_samples})")
    print("=" * 65)
    print(f"{'Metric':<26} | {'Raw Softmax':>13} | {'Temperature Scaled':>19}")
    print("-" * 65)
    print(f"{'Expected Calib Error (ECE)':<26} | {raw_ece*100:>12.2f}% | {calib_ece*100:>18.2f}%")
    print(f"{'Negative Log-Likelihood':<26} | {raw_nll:>13.4f} | {calib_nll:>19.4f}")
    print(f"{'Brier Score':<26} | {raw_brier:>13.4f} | {calib_brier:>19.4f}")
    print(f"{'Accuracy':<26} | {raw_acc*100:>12.2f}% | {calib_acc*100:>18.2f}%")
    print(f"{'Macro Precision':<26} | {raw_prec*100:>12.2f}% | {calib_prec*100:>18.2f}%")
    print(f"{'Macro Recall':<26} | {raw_rec*100:>12.2f}% | {calib_rec*100:>18.2f}%")
    print(f"{'Macro F1':<26} | {raw_f1*100:>12.2f}% | {calib_f1*100:>18.2f}%")
    print(f"{'ROC-AUC':<26} | {raw_auc*100:>12.2f}% | {calib_auc*100:>18.2f}%")
    print("=" * 65)

    correct = calib_preds == val_labels_np
    print(f"\n  Mean entropy (correct)  : {np.mean(calib_entropy[correct]):.4f} bits")
    print(f"  Mean entropy (incorrect): {np.mean(calib_entropy[~correct]):.4f} bits"
          if (~correct).any() else "  Mean entropy (incorrect): n/a (no errors)")

    # 9. Reliability diagrams
    plot_reliability_diagram(
        raw_diag, raw_ece, "CF Classifier - Raw Softmax", os.path.join(PLOT_DIR, "reliability_raw.png")
    )
    plot_reliability_diagram(
        calib_diag, calib_ece,
        f"CF Classifier - Temperature-Scaled (T={best_T:.2f})",
        os.path.join(PLOT_DIR, "reliability_calibrated.png"),
    )
    print(f"\n  Reliability diagrams -> {PLOT_DIR}")

    # 10. Report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(
            f"""# Phase 5 Calibration Report — Community Forensics-Small

## Methodology
- Backbone: DINOv2-Small (ViT-S/14), frozen
- Head: `models/dinov2_authenticity_head_cf.pth`
- Calibration samples: {n_samples} (validation split ONLY)
- Test split: NOT read during calibration
- Learned temperature: **T = {best_T:.4f}**
- Class mapping: `{ckpt["class_to_idx"]}`

## Validation Metrics (N={n_samples})

| Metric | Raw Softmax | Temperature Scaled |
| :--- | ---: | ---: |
| ECE | {raw_ece*100:.2f}% | {calib_ece*100:.2f}% |
| NLL | {raw_nll:.4f} | {calib_nll:.4f} |
| Brier | {raw_brier:.4f} | {calib_brier:.4f} |
| Accuracy | {raw_acc*100:.2f}% | {calib_acc*100:.2f}% |
| Macro F1 | {raw_f1*100:.2f}% | {calib_f1*100:.2f}% |
| ROC-AUC | {raw_auc*100:.2f}% | {calib_auc*100:.2f}% |

Note: Temperature scaling is rank-preserving, so accuracy, F1 and ROC-AUC
are mathematically invariant. Only probability alignment (ECE/NLL/Brier) changes.

## Artifacts
- `{CALIB_OUT}`
- `{PLOT_DIR}/reliability_raw.png`
- `{PLOT_DIR}/reliability_calibrated.png`

## Scope Limitation
Metrics above are validation-split metrics. Held-out test evaluation is a
separate step and was deliberately not performed here.
"""
        )
    print(f"  Report -> {REPORT_PATH}")
    print("=" * 70)
    print(f"  CF CALIBRATION COMPLETE — T = {best_T:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()