"""
Pre-Training Dataset Validator for Community Forensics-Small Dataset.

Verifies dataset integrity, class mapping, image readability, resolution
statistics, duplicate detection, and DINOv2 compatibility BEFORE training.

Does NOT modify any files. Read-only validation.

Usage:
    python -m app.validate_cf_dataset
"""

import os
import json
import hashlib
from PIL import Image
from collections import Counter

import torch

from app.models_vfm import load_dinov2_small, get_dinov2_transforms, EMBEDDING_DIM


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join("dataset", "cf_small")
CLASSES = ["ai_edited", "authentic"]
SPLITS = ["train", "val", "test"]

EXPECTED_COUNTS = {
    ("train", "authentic"): 700,
    ("train", "ai_edited"): 700,
    ("val", "authentic"): 100,
    ("val", "ai_edited"): 100,
    ("test", "authentic"): 200,
    ("test", "ai_edited"): 200,
}

EXPECTED_CLASS_MAPPING = {"ai_edited": 0, "authentic": 1}

SAMPLES_PER_SPLIT = 5  # Number of images to test for DINOv2 compatibility


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def get_image_files(directory):
    """Return sorted list of image filenames in directory."""
    if not os.path.isdir(directory):
        return []
    return sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])


def compute_md5(filepath):
    """Compute MD5 hash of file content."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def print_header(text):
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def print_pass(msg):
    print(f"  [PASS] {msg}")


def print_fail(msg):
    print(f"  [FAIL] {msg}")


def print_info(msg):
    print(f"  [INFO] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Validation Checks
# ─────────────────────────────────────────────────────────────────────────────
def check_counts():
    """Check 1: Exact counts per split/class."""
    print_header("CHECK 1: EXACT COUNTS")
    all_pass = True
    for (split, class_name), expected in EXPECTED_COUNTS.items():
        directory = os.path.join(BASE_DIR, split, class_name)
        files = get_image_files(directory)
        actual = len(files)
        if actual == expected:
            print_pass(f"{split}/{class_name}: {actual} (expected {expected})")
        else:
            print_fail(f"{split}/{class_name}: {actual} (expected {expected})")
            all_pass = False
    return all_pass


def check_readable_and_rgb():
    """Check 2 & 3: Every image is readable by PIL and is RGB."""
    print_header("CHECK 2 & 3: IMAGE READABILITY AND RGB MODE")
    all_pass = True
    total = 0
    errors = []

    for split in SPLITS:
        for class_name in CLASSES:
            directory = os.path.join(BASE_DIR, split, class_name)
            files = get_image_files(directory)
            for filename in files:
                filepath = os.path.join(directory, filename)
                total += 1
                try:
                    img = Image.open(filepath)
                    img.verify()  # Verify it's a valid image
                    img = Image.open(filepath)  # Reopen after verify
                    if img.mode != "RGB":
                        errors.append(f"{split}/{class_name}/{filename}: mode={img.mode}")
                        all_pass = False
                except Exception as e:
                    errors.append(f"{split}/{class_name}/{filename}: {e}")
                    all_pass = False

    if all_pass:
        print_pass(f"All {total} images are readable and RGB")
    else:
        print_fail(f"{len(errors)} images have issues:")
        for err in errors[:10]:  # Show first 10 errors
            print(f"    {err}")
    return all_pass


def check_resolution_stats():
    """Check 4: Resolution statistics per split/class."""
    print_header("CHECK 4: RESOLUTION STATISTICS")
    for split in SPLITS:
        for class_name in CLASSES:
            directory = os.path.join(BASE_DIR, split, class_name)
            files = get_image_files(directory)
            resolutions = []
            for filename in files:
                filepath = os.path.join(directory, filename)
                try:
                    img = Image.open(filepath)
                    resolutions.append(img.size)
                except Exception:
                    pass

            if resolutions:
                widths = [r[0] for r in resolutions]
                heights = [r[1] for r in resolutions]
                unique_res = sorted(set(resolutions))
                print_info(f"{split}/{class_name}:")
                print(f"    Count: {len(resolutions)}")
                print(f"    Min resolution: {min(widths)}x{min(heights)}")
                print(f"    Max resolution: {max(widths)}x{max(heights)}")
                print(f"    Unique resolutions: {len(unique_res)}")
                if len(unique_res) <= 10:
                    print(f"    All resolutions: {unique_res}")
                else:
                    print(f"    Sample resolutions: {unique_res[:5]} ... {unique_res[-5:]}")
    return True


def check_duplicate_filenames():
    """Check 5: No duplicate filename within each class across splits."""
    print_header("CHECK 5: DUPLICATE FILENAMES (within class)")
    all_pass = True
    for class_name in CLASSES:
        class_files = {}
        for split in SPLITS:
            directory = os.path.join(BASE_DIR, split, class_name)
            files = get_image_files(directory)
            for f in files:
                if f in class_files:
                    print_fail(
                        f"Duplicate filename '{f}' in "
                        f"'{class_files[f]}' and '{split}/{class_name}'"
                    )
                    all_pass = False
                else:
                    class_files[f] = f"{split}/{class_name}"
    if all_pass:
        print_pass("No duplicate filenames within any class across splits")
    return all_pass


def check_duplicate_content():
    """Check 6: No duplicate image content within each class across splits."""
    print_header("CHECK 6: DUPLICATE CONTENT (MD5 hash, within class)")
    all_pass = True
    for class_name in CLASSES:
        hashes = {}
        duplicates_found = 0
        for split in SPLITS:
            directory = os.path.join(BASE_DIR, split, class_name)
            files = get_image_files(directory)
            for f in files:
                filepath = os.path.join(directory, f)
                file_hash = compute_md5(filepath)
                if file_hash in hashes:
                    print_fail(
                        f"Duplicate content (MD5: {file_hash[:12]}...) in "
                        f"'{hashes[file_hash]}' and '{split}/{class_name}/{f}'"
                    )
                    duplicates_found += 1
                    all_pass = False
                else:
                    hashes[file_hash] = f"{split}/{class_name}/{f}"
        if duplicates_found == 0:
            print_pass(f"{class_name}: {len(hashes)} unique images across all splits")
    return all_pass


def check_class_mapping():
    """Check 7: Verify class mapping matches ImageFolder alphabetical order."""
    print_header("CHECK 7: CLASS MAPPING")
    # ImageFolder sorts classes alphabetically
    sorted_classes = sorted(CLASSES)
    actual_mapping = {c: i for i, c in enumerate(sorted_classes)}

    all_pass = True
    for class_name in CLASSES:
        expected = EXPECTED_CLASS_MAPPING[class_name]
        actual = actual_mapping[class_name]
        if actual == expected:
            print_pass(f"{class_name} = {actual}")
        else:
            print_fail(f"{class_name} = {actual} (expected {expected})")
            all_pass = False

    print_info(f"ImageFolder will produce: {actual_mapping}")
    return all_pass


def check_dinov2_compatibility():
    """Check 8 & 9: DINOv2 preprocessing and backbone compatibility."""
    print_header("CHECK 8 & 9: DINOv2 COMPATIBILITY")
    all_pass = True

    try:
        device = torch.device("cpu")
        print_info("Loading DINOv2-Small backbone...")
        backbone, metadata = load_dinov2_small(device=device)
        print_pass(f"DINOv2-Small loaded: {metadata['model_name']}")
        print_info(f"  Embedding dim: {metadata['embedding_dim']}")
        print_info(f"  Patch size: {metadata['patch_size']}")
        print_info(f"  Expected patch tokens: {metadata['expected_patch_tokens']}")

        if int(metadata["embedding_dim"]) != EMBEDDING_DIM:
            print_fail(f"Embedding dim mismatch: {metadata['embedding_dim']} != {EMBEDDING_DIM}")
            all_pass = False
        else:
            print_pass(f"Embedding dim = {EMBEDDING_DIM}")

        # Test preprocessing on sample images
        transform = get_dinov2_transforms()
        print_info(f"Testing preprocessing on {SAMPLES_PER_SPLIT} samples per split...")

        for split in SPLITS:
            for class_name in CLASSES:
                directory = os.path.join(BASE_DIR, split, class_name)
                files = get_image_files(directory)
                sample_files = files[:SAMPLES_PER_SPLIT]

                for filename in sample_files:
                    filepath = os.path.join(directory, filename)
                    try:
                        img = Image.open(filepath).convert("RGB")
                        tensor = transform(img)
                        if tensor.shape != (3, 224, 224):
                            print_fail(
                                f"{split}/{class_name}/{filename}: "
                                f"tensor shape {tensor.shape} != (3, 224, 224)"
                            )
                            all_pass = False
                    except Exception as e:
                        print_fail(
                            f"{split}/{class_name}/{filename}: preprocessing failed: {e}"
                        )
                        all_pass = False

        print_pass("All sample images preprocess to (3, 224, 224)")

        # Test backbone feature extraction on a few samples
        print_info("Testing backbone feature extraction...")
        test_count = 0
        for split in SPLITS:
            for class_name in CLASSES:
                directory = os.path.join(BASE_DIR, split, class_name)
                files = get_image_files(directory)
                sample_files = files[:2]  # 2 samples per split/class

                for filename in sample_files:
                    filepath = os.path.join(directory, filename)
                    try:
                        img = Image.open(filepath).convert("RGB")
                        tensor = transform(img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            feat_dict = backbone.forward_features(tensor)
                            cls_token = feat_dict["x_norm_clstoken"]
                            if cls_token.shape[1] != EMBEDDING_DIM:
                                print_fail(
                                    f"{split}/{class_name}/{filename}: "
                                    f"CLS dim {cls_token.shape[1]} != {EMBEDDING_DIM}"
                                )
                                all_pass = False
                            test_count += 1
                    except Exception as e:
                        print_fail(
                            f"{split}/{class_name}/{filename}: backbone failed: {e}"
                        )
                        all_pass = False

        print_pass(f"Backbone produced {EMBEDDING_DIM}-D CLS features from {test_count} samples")

    except Exception as e:
        print_fail(f"DINOv2 compatibility check failed: {e}")
        all_pass = False

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  COMMUNITY FORENSICS-SMALL DATASET VALIDATOR")
    print("=" * 70)
    print(f"  Base directory: {BASE_DIR}")
    print(f"  Classes: {CLASSES}")
    print(f"  Splits: {SPLITS}")

    results = {}

    results["counts"] = check_counts()
    results["readable_rgb"] = check_readable_and_rgb()
    results["resolution_stats"] = check_resolution_stats()
    results["duplicate_filenames"] = check_duplicate_filenames()
    results["duplicate_content"] = check_duplicate_content()
    results["class_mapping"] = check_class_mapping()
    results["dinov2_compatibility"] = check_dinov2_compatibility()

    # Summary
    print_header("VALIDATION SUMMARY")
    all_pass = all(results.values())
    for check_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_name}")

    print()
    if all_pass:
        print("  ALL CHECKS PASSED — dataset is ready for training.")
    else:
        print("  SOME CHECKS FAILED — fix issues before training.")

    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
