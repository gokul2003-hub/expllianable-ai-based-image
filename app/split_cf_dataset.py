"""
Deterministic Train/Validation Split for Community Forensics-Small Dataset.

MOVES exactly 100 images per class from dataset/cf_small/train/ into
dataset/cf_small/val/ to create a stratified validation split.

Final counts:
  train/authentic  = 700
  train/ai_edited  = 700
  val/authentic    = 100
  val/ai_edited    = 100
  test/authentic   = 200  (untouched)
  test/ai_edited   = 200  (untouched)

Uses random.seed(42) for reproducibility.
Does NOT modify test split.
Does NOT copy — only MOVES files.
"""

import os
import json
import random
import shutil
import hashlib

SEED = 42
VAL_PER_CLASS = 100

BASE_DIR = os.path.join("dataset", "cf_small")
CLASSES = ["ai_edited", "authentic"]
SPLITS = ["train", "val", "test"]


def main():
    random.seed(SEED)

    manifest = {
        "seed": SEED,
        "val_per_class": VAL_PER_CLASS,
        "class_mapping": {},
        "splits": {},
    }

    for class_name in CLASSES:
        train_dir = os.path.join(BASE_DIR, "train", class_name)
        val_dir = os.path.join(BASE_DIR, "val", class_name)
        test_dir = os.path.join(BASE_DIR, "test", class_name)

        # Ensure val directory exists
        os.makedirs(val_dir, exist_ok=True)

        # List current train files
        train_files = sorted([
            f for f in os.listdir(train_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        total_train = len(train_files)
        expected_total = 700 + VAL_PER_CLASS  # 800

        if total_train != expected_total:
            raise ValueError(
                f"Expected {expected_total} files in {train_dir}, found {total_train}"
            )

        # Deterministic shuffle
        random.shuffle(train_files)

        # Split: first 700 stay, last 100 move to val
        stay_files = train_files[:700]
        move_files = train_files[700:]

        if len(move_files) != VAL_PER_CLASS:
            raise ValueError(
                f"Expected to move {VAL_PER_CLASS} files, got {len(move_files)}"
            )

        # MOVE files from train to val
        for filename in move_files:
            src = os.path.join(train_dir, filename)
            dst = os.path.join(val_dir, filename)
            shutil.move(src, dst)

        # Record manifest
        manifest["splits"][f"train/{class_name}"] = sorted(stay_files)
        manifest["splits"][f"val/{class_name}"] = sorted(move_files)

        # Class mapping (ImageFolder alphabetical order)
        manifest["class_mapping"][class_name] = CLASSES.index(class_name)

        print(f"[{class_name}] train: {len(stay_files)}, val: {len(move_files)}")

    # Record test files (untouched)
    for class_name in CLASSES:
        test_dir = os.path.join(BASE_DIR, "test", class_name)
        test_files = sorted([
            f for f in os.listdir(test_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        manifest["splits"][f"test/{class_name}"] = test_files
        print(f"[{class_name}] test: {len(test_files)} (untouched)")

    # Verify no duplicates WITHIN each class across train/val/test
    for class_name in CLASSES:
        class_files = {}
        for split in SPLITS:
            split_key = f"{split}/{class_name}"
            for f in manifest["splits"][split_key]:
                if f in class_files:
                    raise ValueError(
                        f"Duplicate filename '{f}' found in both "
                        f"'{class_files[f]}' and '{split_key}'"
                    )
                class_files[f] = split_key

    # Stronger check: detect identical file CONTENT within each class
    # (catches same image with different filenames across splits)
    for class_name in CLASSES:
        hashes = {}
        for split in SPLITS:
            split_dir = os.path.join(BASE_DIR, split, class_name)
            for f in manifest["splits"][f"{split}/{class_name}"]:
                filepath = os.path.join(split_dir, f)
                with open(filepath, "rb") as fh:
                    file_hash = hashlib.md5(fh.read()).hexdigest()
                if file_hash in hashes:
                    raise ValueError(
                        f"Duplicate file content (MD5: {file_hash}) found in both "
                        f"'{hashes[file_hash]}' and '{split}/{class_name}/{f}'"
                    )
                hashes[file_hash] = f"{split}/{class_name}/{f}"

    # Save manifest
    os.makedirs("outputs", exist_ok=True)
    manifest_path = os.path.join("outputs", "cf_split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSplit manifest saved to: {manifest_path}")
    print(f"Total unique files across all splits: {len(all_files)}")
    print("\nFinal counts:")
    print(f"  train/authentic: {len(manifest['splits']['train/authentic'])}")
    print(f"  train/ai_edited: {len(manifest['splits']['train/ai_edited'])}")
    print(f"  val/authentic:   {len(manifest['splits']['val/authentic'])}")
    print(f"  val/ai_edited:   {len(manifest['splits']['val/ai_edited'])}")
    print(f"  test/authentic:  {len(manifest['splits']['test/authentic'])}")
    print(f"  test/ai_edited:  {len(manifest['splits']['test/ai_edited'])}")


if __name__ == "__main__":
    main()
