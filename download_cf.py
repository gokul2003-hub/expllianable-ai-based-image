import os
import requests
import ast
from PIL import Image
from io import BytesIO

BASE = "https://huggingface.co/datasets/OwensLab/CommunityForensics-Small/resolve/main/data"

OUT = "dataset/cf_small"

TARGET_TRAIN = 800
TARGET_TEST = 200


def download_shard(shard, label, train_count, test_count):
    url = f"{BASE}/HFCF_small_{shard}.parquet"
    temp = f"cf_shard_{shard}.parquet"

    print(f"\nDownloading shard {shard}...")
    print(url)

    r = requests.get(url, timeout=600)

    if r.status_code != 200:
        raise RuntimeError(f"Download failed: HTTP {r.status_code}")

    with open(temp, "wb") as f:
        f.write(r.content)

    print("Downloaded:", len(r.content) / (1024 * 1024), "MB")

    import pyarrow.parquet as pq

    table = pq.read_table(
        temp,
        columns=["image_data", "label"]
    )

    rows = table.to_pylist()

    train_folder = (
        f"{OUT}/train/authentic"
        if label == 0
        else f"{OUT}/train/ai_edited"
    )

    test_folder = (
        f"{OUT}/test/authentic"
        if label == 0
        else f"{OUT}/test/ai_edited"
    )

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    total_needed = train_count + test_count
    saved = 0

    for row in rows:

        if saved >= total_needed:
            break

        raw = row["image_data"]

        if isinstance(raw, str):
            raw = ast.literal_eval(raw)

        img = Image.open(BytesIO(raw)).convert("RGB")

        if saved < train_count:
            folder = train_folder
            prefix = "train"
            number = saved
        else:
            folder = test_folder
            prefix = "test"
            number = saved - train_count

        path = os.path.join(folder, f"{prefix}_{number:04d}.png")
        img.save(path)

        saved += 1

        if saved % 50 == 0:
            print(
                f"label={label} | saved={saved}/{total_needed} | "
                f"resolution={img.size}"
            )

    os.remove(temp)

    print(f"DONE shard {shard}: {saved} images")


# Fake images
download_shard(
    shard=0,
    label=1,
    train_count=TARGET_TRAIN,
    test_count=TARGET_TEST
)

# Real images
download_shard(
    shard=180,
    label=0,
    train_count=TARGET_TRAIN,
    test_count=TARGET_TEST
)

print("\n================================")
print("DATASET CREATION COMPLETE")
print("================================")
print("Train authentic :", TARGET_TRAIN)
print("Train AI-edited :", TARGET_TRAIN)
print("Test authentic  :", TARGET_TEST)
print("Test AI-edited  :", TARGET_TEST)
print("Total           :", 2000)