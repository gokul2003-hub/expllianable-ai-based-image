"""
FINAL Community Forensics-Small Benchmark — Held-Out Test Split (400 images).

Evaluates the trained CF head + frozen DINOv2-Small backbone on the strictly
held-out test split. Uses the ALREADY-FITTED calibration temperature and the
ALREADY-FITTED OOD parameters. Does not refit, retrain, or recalibrate anything.

Components:
  1. Classification  (accuracy, macro P/R/F1, ROC-AUC, confusion matrix, per-class)
  2. Calibration     (ECE, NLL, Brier) using calibration_params_cf.json
  3. OOD             (distance/threshold behavior) using ood_params_cf.json
  4. TTA             (horizontal-flip agreement) using existing methodology

Outputs -> outputs/benchmark_cf/
"""

import os
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

from app.models_vfm import (
    load_dinov2_small,
    get_dinov2_transforms,
    DINOv2ClassificationHead,
    DINOv2AuthenticityClassifier,
    EMBEDDING_DIM,
)
from app.calibration import compute_ece, compute_nll, compute_brier_score, compute_entropy
from app.ood_detection import DINOv2CentroidOODDetector


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
TEST_DIR = os.path.join("dataset", "cf_small", "test")
BACKBONE_PTH = os.path.join("models", "dinov2_vits14_pretrain.pth")
HEAD_CKPT = os.path.join("models", "dinov2_authenticity_head_cf.pth")
CALIB_JSON = os.path.join("models", "calibration_params_cf.json")
OOD_JSON = os.path.join("models", "ood_params_cf.json")
OUT_DIR = os.path.join("outputs", "benchmark_cf")
REPORT_PATH = os.path.join("outputs", "benchmark_cf_report.md")

BATCH_SIZE = 32
EXPECTED_TEST = 400


# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cpu")

    print("=" * 70)
    print("  FINAL CF BENCHMARK — HELD-OUT TEST SPLIT")
    print("=" * 70)

    # ── Load model ────────────────────────────────────────────────────────────
    backbone, _ = load_dinov2_small(device=device)
    ckpt = torch.load(HEAD_CKPT, map_location=device, weights_only=False)
    head = DINOv2ClassificationHead(
        in_features=EMBEDDING_DIM,
        hidden_dim=ckpt["hidden_dim"],
        num_classes=len(ckpt["classes"]),
    ).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    full_model = DINOv2AuthenticityClassifier(backbone, head).to(device)
    full_model.eval()
    classes = ckpt["classes"]
    print(f"  Head: {HEAD_CKPT}")
    print(f"  Classes: {classes}  mapping: {ckpt['class_to_idx']}")

    # ── Load calibration ───────────────────────────────────────────────────────
    with open(CALIB_JSON, "r", encoding="utf-8") as f:
        calib = json.load(f)
    temperature = float(calib["temperature"])
    print(f"  Calibration T = {temperature:.4f} (pre-fitted, NOT refit)")

    # ── Load OOD ──────────────────────────────────────────────────────────────
    ood_available = os.path.exists(OOD_JSON)
    ood_params = None
    if ood_available:
        with open(OOD_JSON, "r", encoding="utf-8") as f:
            ood_params = json.load(f)
        print(f"  OOD params: {OOD_JSON} (pre-fitted, NOT refit)")
    else:
        print(f"  OOD params: {OOD_JSON} — *** NOT FOUND, OOD EVALUATION SKIPPED ***")

    # ── Test data ─────────────────────────────────────────────────────────────
    transform = get_dinov2_transforms()
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    n_test = len(test_dataset)
    print(f"\n  Test images: {n_test} (expected {EXPECTED_TEST})")
    assert n_test == EXPECTED_TEST, f"Expected {EXPECTED_TEST} test images, got {n_test}"
    print(f"  Class mapping: {test_dataset.class_to_idx}")

    # ── Single-crop inference ─────────────────────────────────────────────────
    print("\n[1/4] Single-crop inference...")
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = full_model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels.clone())
    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    # Calibrated probabilities
    cal_logits = logits / temperature
    exp_c = np.exp(cal_logits - np.max(cal_logits, axis=-1, keepdims=True))
    cal_probs = exp_c / np.sum(exp_c, axis=-1, keepdims=True)
    preds = np.argmax(cal_probs, axis=1)

    # ── 1. Classification metrics ─────────────────────────────────────────────
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    auc = roc_auc_score(labels, cal_probs[:, 1])
    cm = confusion_matrix(labels, preds)

    # Per-class
    per_class = {}
    for i, cname in enumerate(classes):
        mask = labels == i
        pc = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0
        )
        per_class[cname] = {
            "precision": float(pc[0][i]),
            "recall": float(pc[1][i]),
            "f1": float(pc[2][i]),
            "support": int(mask.sum()),
        }

    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Macro P:   {prec*100:.2f}%")
    print(f"  Macro R:   {rec*100:.2f}%")
    print(f"  Macro F1:  {f1*100:.2f}%")
    print(f"  ROC-AUC:   {auc*100:.2f}%")
    print(f"  Confusion matrix:\n{cm}")

    # ── 2. Calibration metrics ────────────────────────────────────────────────
    print("\n[2/4] Calibration metrics (pre-fitted T)...")
    ece, _ = compute_ece(cal_probs, labels, n_bins=15)
    nll = compute_nll(cal_probs, labels)
    brier = compute_brier_score(cal_probs, labels)
    entropy = compute_entropy(cal_probs)
    print(f"  ECE:    {ece*100:.2f}%")
    print(f"  NLL:    {nll:.4f}")
    print(f"  Brier:  {brier:.4f}")
    print(f"  Mean entropy: {np.mean(entropy):.4f} bits")

    # ── 3. OOD evaluation ─────────────────────────────────────────────────────
    ood_results = {"status": "SKIPPED — ood_params_cf.json not found"}
    if ood_params is not None:
        print("\n[3/4] OOD evaluation (pre-fitted threshold)...")
        detector = DINOv2CentroidOODDetector(metric=ood_params.get("metric", "cosine"))
        detector.classes = ood_params["classes"]
        detector.centroids = {
            "authentic": np.array(ood_params["centroid_authentic"], dtype=np.float32),
            "ai_edited": np.array(ood_params["centroid_synthetic"], dtype=np.float32),
        }
        detector.threshold_95 = float(ood_params["threshold_95"])

        # Extract test features
        test_feats = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                fd = backbone.forward_features(images)
                test_feats.append(fd["x_norm_clstoken"].cpu())
        test_feats_np = torch.cat(test_feats, dim=0).numpy()

        is_ood, scores = detector.predict_ood(test_feats_np)
        ood_results = {
            "status": "COMPLETED",
            "threshold_95": float(ood_params["threshold_95"]),
            "mean_distance": float(np.mean(scores)),
            "std_distance": float(np.std(scores)),
            "flagged_ood_count": int(np.sum(is_ood)),
            "flagged_ood_rate": float(np.mean(is_ood)),
        }
        print(f"  Threshold: {ood_params['threshold_95']:.4f}")
        print(f"  Mean distance: {np.mean(scores):.4f}")
        print(f"  Flagged OOD: {np.sum(is_ood)}/{len(scores)} ({np.mean(is_ood)*100:.2f}%)")
    else:
        print("\n[3/4] OOD evaluation SKIPPED (no pre-fitted artifact)")

    # ── 4. TTA ────────────────────────────────────────────────────────────────
    print("\n[4/4] TTA (horizontal flip)...")
    flip_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    flip_loader = DataLoader(flip_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    flip_logits = []
    with torch.no_grad():
        for images, _ in flip_loader:
            images = images.to(device)
            flipped = torch.flip(images, dims=[3])
            flogits = full_model(flipped)
            flip_logits.append(flogits.cpu())
    flip_logits = torch.cat(flip_logits, dim=0).numpy()

    flip_cal = flip_logits / temperature
    exp_f = np.exp(flip_cal - np.max(flip_cal, axis=-1, keepdims=True))
    flip_probs = exp_f / np.sum(exp_f, axis=-1, keepdims=True)

    tta_probs = 0.5 * (cal_probs + flip_probs)
    tta_preds = np.argmax(tta_probs, axis=1)
    tta_acc = accuracy_score(labels, tta_preds)
    agree = preds == tta_preds
    agreement_rate = np.mean(agree)
    delta = (tta_preds.astype(int) - preds.astype(int))
    n_disagree = int(np.sum(~agree))

    print(f"  Single-crop accuracy: {acc*100:.2f}%")
    print(f"  TTA accuracy:        {tta_acc*100:.2f}%")
    print(f"  Agreement:           {agreement_rate*100:.2f}%")
    print(f"  Disagreements:       {n_disagree}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    metrics = {
        "dataset": "cf_small",
        "split": "test",
        "n_test": n_test,
        "classes": classes,
        "class_to_idx": ckpt["class_to_idx"],
        "classification": {
            "accuracy": float(acc),
            "macro_precision": float(prec),
            "macro_recall": float(rec),
            "macro_f1": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
        },
        "calibration": {
            "temperature": temperature,
            "ece": float(ece),
            "nll": float(nll),
            "brier": float(brier),
            "mean_entropy": float(np.mean(entropy)),
        },
        "ood": ood_results,
        "tta": {
            "single_crop_accuracy": float(acc),
            "tta_accuracy": float(tta_acc),
            "agreement_rate": float(agreement_rate),
            "n_disagreements": n_disagree,
        },
    }

    metrics_path = os.path.join(OUT_DIR, "benchmark_cf_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved: {metrics_path}")

    # Confusion matrix plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=20)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("CF Test Confusion Matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
    fig.tight_layout()
    cm_path = os.path.join(OUT_DIR, "confusion_matrix_cf.png")
    fig.savefig(cm_path, dpi=200)
    plt.close()
    print(f"  Confusion matrix: {cm_path}")

    # Report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"""# FINAL CF Benchmark Report — Held-Out Test Split

## Setup
- Backbone: DINOv2-Small (ViT-S/14), frozen
- Head: `models/dinov2_authenticity_head_cf.pth`
- Calibration: `models/calibration_params_cf.json` (T = {temperature:.4f}, pre-fitted)
- OOD: `models/ood_params_cf.json` — {"USED" if ood_available else "NOT AVAILABLE"}
- Test split: `dataset/cf_small/test` — {n_test} images (200 authentic + 200 ai_edited)

## 1. Classification (Test, N={n_test})

| Metric | Value |
|--------|-------|
| Accuracy | {acc*100:.2f}% |
| Macro Precision | {prec*100:.2f}% |
| Macro Recall | {rec*100:.2f}% |
| Macro F1 | {f1*100:.2f}% |
| ROC-AUC | {auc*100:.2f}% |

Confusion Matrix:
```
{cm}
```

Per-class:
""")
        for cname, vals in per_class.items():
            f.write(f"- **{cname}**: P={vals['precision']*100:.2f}%, R={vals['recall']*100:.2f}%, F1={vals['f1']*100:.2f}%, support={vals['support']}\n")

        f.write(f"""
## 2. Calibration (Test, pre-fitted T={temperature:.4f})

| Metric | Value |
|--------|-------|
| ECE | {ece*100:.2f}% |
| NLL | {nll:.4f} |
| Brier | {brier:.4f} |
| Mean Entropy | {np.mean(entropy):.4f} bits |

## 3. OOD

{json.dumps(ood_results, indent=2)}

## 4. TTA (Horizontal Flip)

| Metric | Value |
|--------|-------|
| Single-crop accuracy | {acc*100:.2f}% |
| TTA accuracy | {tta_acc*100:.2f}% |
| Agreement rate | {agreement_rate*100:.2f}% |
| Disagreements | {n_disagree} |

## Validation vs Test Distinction

- **Validation (N=200)**: used for checkpoint selection and calibration fitting only.
- **Test (N=400)**: strictly held out; first and only evaluation in this report.
- No test data was used for training, calibration, OOD thresholding, or model selection.

## Artifacts
- `{metrics_path}`
- `{cm_path}`
- `{REPORT_PATH}`
""")
    print(f"  Report: {REPORT_PATH}")
    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
