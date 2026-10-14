from datasets import load_dataset
from PIL import Image
from io import BytesIO
import ast
import os

TARGET = 800
OUT = "dataset/cf_small"

def extract(label, shard, folder):
    print(f"START label={label}, shard={shard}")

    ds = load_dataset(
        "parquet",
        data_files=f"https://huggingface.co/datasets/OwensLab/CommunityForensics-Small/resolve/main/data/HFCF_small_{shard}.parquet",
        split="train",
        streaming=True,
        columns=["image_data", "label"],
        filters=[("label", "=", label)],
    )

    os.makedirs(folder, exist_ok=True)

    count = 0

    for row in ds:
        raw = row["image_data"]

        if isinstance(raw, str):
            raw = ast.literal_eval(raw)

        img = Image.open(BytesIO(raw)).convert("RGB")
        img.save(os.path.join(folder, f"{count:04d}.png"))

        count += 1

        if count % 50 == 0:
            print("SAVED:", count, img.size)

        if count >= TARGET:
            break

    print("DONE:", label, count)


# Fake shard
extract(
    1,
    0,
    f"{OUT}/train/ai_edited"
)

# Real shard
extract(
    0,
    180,
    f"{OUT}/train/authentic"
)

print("ALL DONE")