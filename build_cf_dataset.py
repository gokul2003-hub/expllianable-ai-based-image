from datasets import load_dataset
from PIL import Image
from io import BytesIO
import os

OUT = "dataset/cf_small"
TARGET = {0: 800, 1: 800}
count = {0: 0, 1: 0}

os.makedirs(f"{OUT}/train/authentic", exist_ok=True)
os.makedirs(f"{OUT}/train/ai_edited", exist_ok=True)

ds = load_dataset(
    "OwensLab/CommunityForensics-Small",
    split="train",
    streaming=True
).shuffle(seed=42, buffer_size=2000)

for row in ds:
    label = int(row["label"])

    if count[label] >= TARGET[label]:
        if count[0] >= 800 and count[1] >= 800:
            break
        continue

    img = Image.open(BytesIO(row["image_data"])).convert("RGB")

    if label == 0:
        folder = f"{OUT}/train/authentic"
    else:
        folder = f"{OUT}/train/ai_edited"

    filename = f"{folder}/{count[label]:04d}.png"
    img.save(filename)

    count[label] += 1
    print("SAVED", label, count[label], img.size)

print("DONE:", count)
