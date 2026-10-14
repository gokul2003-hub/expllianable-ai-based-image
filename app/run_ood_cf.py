"""
CF-Specific OOD Calibration — Community Forensics-Small.

Reuses the existing DINOv2CentroidOODDetector methodology from app/ood_detection.py.
Fits class centroids on TRAIN features only. Sets threshold from VALIDATION only.
Test split is NOT used for fitting or threshold selection.

Output: models/ood_params_cf.json
"""

import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from app.models_vfm import load_dinov2_small, get_dinov2_transforms
from app.ood_detection import DINOv2CentroidOODDetector


TRAIN_DIR = os.path.join("dataset", "cf_small", "train")
VAL_DIR = os.path.join("dataset", "cf_small", "val")
# TEST_DIR is intentionally never referenced.

OOD_OUT = os.path.join("models", "ood_params_cf.json")
BATCH_SIZE = 32


def extract_features(split_dir, backbone, transform, device):
    """Extract 384-D CLS features from a split."""
    dataset = datasets.ImageFolder(split_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    feats, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            fd = backbone.forward_features(images)
            feats.append(fd["x_norm_clstoken"].cpu())
            labels.append(batch_labels.clone())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0), dataset.classes


def main():
    print("=" * 70)
    print("  CF OOD CALIBRATION — CENTROID FITTING + THRESHOLD")
    print("=" * 70)

    device = torch.device("cpu")
    os.makedirs("models", exist_ok=True)

    # Load frozen backbone
    print("\n[1/4] Loading DINOv2-Small backbone (frozen)...")
    backbone, metadata = load_dinov2_small(device=device)
    print(f"  Backbone: {metadata['model_name']}")

    transform = get_dinov2_transforms()

    # Extract train features
    print("\n[2/4] Extracting TRAIN features (centroid fitting)...")
    train_feats, train_labels, classes = extract_features(TRAIN_DIR, backbone, transform, device)
    train_feats_np = train_feats.numpy()
    train_labels_np = train_labels.numpy()
    print(f"  Train features: {train_feats_np.shape}")
    print(f"  Train labels:   {train_labels_np.shape}")
    print(f"  Classes:        {classes}")

    # Extract val features
    print("\n[3/4] Extracting VALIDATION features (threshold calibration)...")
    val_feats, val_labels, _ = extract_features(VAL_DIR, backbone, transform, device)
    val_feats_np = val_feats.numpy()
    val_labels_np = val_labels.numpy()
    print(f"  Val features:   {val_feats_np.shape}")
    print(f"  Val labels:     {val_labels_np.shape}")

    # Fit OOD detector
    print("\n[4/4] Fitting DINOv2CentroidOODDetector (cosine)...")
    detector = DINOv2CentroidOODDetector(metric="cosine")
    detector.fit(train_feats_np, train_labels_np, classes)
    print(f"  Centroids fitted for: {list(detector.centroids.keys())}")

    # Set threshold from validation (95th percentile)
    th_95 = detector.set_threshold_from_val(val_feats_np)
    th_99 = detector.threshold_99
    print(f"  Threshold (95th pct): {th_95:.4f}")
    print(f"  Threshold (99th pct): {th_99:.4f}")

    # Build output JSON preserving existing structure
    ood_params = {
        "metric": "cosine",
        "classes": classes,
        "class_to_idx": {c: i for i, c in enumerate(classes)},
        "threshold_95": float(th_95),
        "threshold_99": float(th_99),
        "centroid_authentic": detector.centroids["authentic"].tolist(),
        "centroid_synthetic": detector.centroids["ai_edited"].tolist(),
        "dataset": "cf_small",
        "train_reference_count": int(len(train_labels_np)),
        "validation_count": int(len(val_labels_np)),
        "embedding_dim": 384,
        "fitted_on": "train",
        "threshold_calibrated_on": "validation",
    }

    with open(OOD_OUT, "w", encoding="utf-8") as f:
        json.dump(ood_params, f, indent=4)
    print(f"\n  Saved: {OOD_OUT}")

    # Verification
    print("\n" + "=" * 70)
    print("  VERIFICATION")
    print("=" * 70)
    with open(OOD_OUT, "r", encoding="utf-8") as f:
        check = json.load(f)
    print(f"  File exists:        True")
    print(f"  Valid JSON:         True")
    print(f"  threshold_95:        {check['threshold_95']:.4f}")
    print(f"  threshold_99:        {check['threshold_99']:.4f}")
    print(f"  metric:              {check['metric']}")
    print(f"  classes:             {check['classes']}")
    print(f"  class_to_idx:        {check['class_to_idx']}")
    print(f"  train count:         {check['train_reference_count']}")
    print(f"  validation count:    {check['validation_count']}")
    print(f"  centroid_authentic:  [{len(check['centroid_authentic'])}-D vector present]")
    print(f"  centroid_synthetic: [{len(check['centroid_synthetic'])}-D vector present]")
    print(f"  dataset:             {check['dataset']}")
    print(f"  test data used:      NO (never read)")
    print("=" * 70)
    print("  CF OOD CALIBRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
