"""
DINOv2 Authenticity Head Training for Community Forensics-Small Dataset.

Trains ONLY the classification head on top of a FROZEN DINOv2-Small backbone.
Uses the existing project architecture and preprocessing.

Dataset:
  dataset/cf_small/train/  (700 authentic + 700 ai_edited = 1,400)
  dataset/cf_small/val/    (100 authentic + 100 ai_edited = 200)
  dataset/cf_small/test/   (NEVER used during training or model selection)

Model:
  DINOv2-Small (ViT-S/14) — FROZEN backbone
  DINOv2ClassificationHead: 384 -> 256 -> 2 (trainable)

Output:
  models/dinov2_authenticity_head_cf.pth

Class mapping (ImageFolder alphabetical):
  ai_edited = 0
  authentic = 1
"""

import os
import copy
import random
import json

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets

from app.models_vfm import (
    load_dinov2_small,
    get_dinov2_transforms,
    DINOv2ClassificationHead,
    EMBEDDING_DIM,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
DATASET_ROOT = os.path.join("dataset", "cf_small")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")
# TEST_DIR is intentionally NOT used — test is evaluation-only

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

# Training hyperparameters (reasonable for 1,400-image training set)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Expected dataset counts (fail clearly if missing)
EXPECTED_TRAIN_PER_CLASS = 700
EXPECTED_VAL_PER_CLASS = 100

# Output artifact (CF-specific, never overwrites old CIFAKE checkpoint)
OUTPUT_CHECKPOINT = os.path.join(MODEL_DIR, "dinov2_authenticity_head_cf.pth")
OUTPUT_HISTORY = os.path.join(OUTPUT_DIR, "training_history_cf.json")


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Validation
# ─────────────────────────────────────────────────────────────────────────────
def validate_dataset():
    """Verify dataset folders exist and have expected counts."""
    print("=" * 70)
    print("  DATASET VALIDATION")
    print("=" * 70)

    errors = []

    for split, expected_per_class in [("train", EXPECTED_TRAIN_PER_CLASS), ("val", EXPECTED_VAL_PER_CLASS)]:
        for class_name in ["ai_edited", "authentic"]:
            directory = os.path.join(DATASET_ROOT, split, class_name)
            if not os.path.isdir(directory):
                errors.append(f"Missing directory: {directory}")
                continue

            files = [
                f for f in os.listdir(directory)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            actual = len(files)
            if actual != expected_per_class:
                errors.append(
                    f"{split}/{class_name}: expected {expected_per_class}, found {actual}"
                )
            else:
                print(f"  [OK] {split}/{class_name}: {actual} images")

    if errors:
        print("\n  DATASET ERRORS:")
        for err in errors:
            print(f"    {err}")
        raise ValueError("Dataset validation failed. Fix issues before training.")

    print("  Dataset validation passed.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Training Functions
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, head, loader, criterion, optimizer, device):
    """Train one epoch on the training split only."""
    head.train()
    # Backbone is always frozen
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # Extract frozen features (no gradient computation for backbone)
        with torch.no_grad():
            feat_dict = model.forward_features(images)
            cls_features = feat_dict["x_norm_clstoken"]  # [B, 384]

        # Train only the classification head
        optimizer.zero_grad()
        outputs = head(cls_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy


def validate(model, head, loader, criterion, device):
    """Evaluate on the validation split only."""
    head.eval()
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            feat_dict = model.forward_features(images)
            cls_features = feat_dict["x_norm_clstoken"]

            outputs = head(cls_features)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  DINOv2 AUTHENTICITY HEAD TRAINING — COMMUNITY FORENSICS-SMALL")
    print("=" * 70)

    # 1. Set seed for reproducibility
    set_seed(SEED)
    print(f"  Random seed: {SEED}")

    # 2. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # 3. Validate dataset
    validate_dataset()

    # 4. Create output directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 5. Load datasets (train and val only — test is NEVER used)
    print("  Loading datasets...")
    transform = get_dinov2_transforms()  # 224x224 preprocessing

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

    print(f"  Train images: {len(train_dataset)}")
    print(f"  Val images:   {len(val_dataset)}")
    print(f"  Classes:      {train_dataset.classes}")
    print(f"  Class mapping: {train_dataset.class_to_idx}")

    # Verify class mapping
    expected_mapping = {"ai_edited": 0, "authentic": 1}
    if train_dataset.class_to_idx != expected_mapping:
        raise ValueError(
            f"Class mapping mismatch: {train_dataset.class_to_idx} != {expected_mapping}"
        )

    # 6. Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # 7. Load FROZEN DINOv2-Small backbone
    print("\n  Loading DINOv2-Small backbone (frozen)...")
    backbone, metadata = load_dinov2_small(device=device)
    print(f"  Backbone: {metadata['model_name']}")
    print(f"  Embedding dim: {metadata['embedding_dim']}")
    print(f"  Backbone is FROZEN — no gradients will be computed for it")

    # 8. Create trainable classification head
    # Architecture: 384 -> 256 -> 2 (preserved from existing project)
    head = DINOv2ClassificationHead(
        in_features=EMBEDDING_DIM,  # 384
        hidden_dim=256,
        num_classes=2,
    ).to(device)
    print(f"  Classification head: {EMBEDDING_DIM} -> 256 -> 2 (trainable)")

    # 9. Loss and optimizer (only head parameters)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")

    # 10. Training loop
    print("\n" + "=" * 70)
    print("  TRAINING")
    print("=" * 70)

    best_val_accuracy = 0.0
    best_head_state = None
    history = []

    for epoch in range(1, EPOCHS + 1):
        # Train on training split only
        train_loss, train_acc = train_one_epoch(
            backbone, head, train_loader, criterion, optimizer, device
        )

        # Validate on validation split only
        val_loss, val_acc = validate(
            backbone, head, val_loader, criterion, device
        )

        # Record history
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

        print(
            f"  Epoch {epoch:2d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
        )

        # Save best checkpoint based on VALIDATION accuracy only
        # Test set is NEVER used for checkpoint selection
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_head_state = copy.deepcopy(head.state_dict())
            print(f"    >>> New best model (val acc: {val_acc*100:.2f}%)")

    # 11. Save best checkpoint
    print("\n" + "=" * 70)
    print("  SAVING CHECKPOINT")
    print("=" * 70)

    if best_head_state is None:
        raise RuntimeError("No best model found — training failed.")

    checkpoint = {
        "head_state_dict": best_head_state,
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "hidden_dim": 256,
        "embedding_dim": EMBEDDING_DIM,
        "best_val_accuracy": best_val_accuracy,
        "epochs_trained": EPOCHS,
        "seed": SEED,
        "dataset": "cf_small",
        "train_counts": {
            "authentic": EXPECTED_TRAIN_PER_CLASS,
            "ai_edited": EXPECTED_TRAIN_PER_CLASS,
        },
        "val_counts": {
            "authentic": EXPECTED_VAL_PER_CLASS,
            "ai_edited": EXPECTED_VAL_PER_CLASS,
        },
    }

    torch.save(checkpoint, OUTPUT_CHECKPOINT)
    print(f"  Checkpoint saved to: {OUTPUT_CHECKPOINT}")
    print(f"  Best validation accuracy: {best_val_accuracy*100:.2f}%")

    # 12. Save training history
    with open(OUTPUT_HISTORY, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"  Training history saved to: {OUTPUT_HISTORY}")

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best validation accuracy: {best_val_accuracy*100:.2f}%")
    print(f"  Checkpoint: {OUTPUT_CHECKPOINT}")
    print(f"  Old CIFAKE checkpoint untouched: models/dinov2_authenticity_head.pth")
    print("=" * 70)


if __name__ == "__main__":
    main()
