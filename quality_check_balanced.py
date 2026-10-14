import requests
import base64
import ast
import os
from PIL import Image
from io import BytesIO

url = "https://datasets-server.huggingface.co/rows?dataset=OwensLab%2FCommunityForensics-Small&config=default&split=train&offset=0&length=100"

data = requests.get(url, timeout=60).json()

os.makedirs("dataset/quality_check/real", exist_ok=True)
os.makedirs("dataset/quality_check/fake", exist_ok=True)

counts = {0: 0, 1: 0}

for item in data["rows"]:
    row = item["row"]
    label = int(row["label"])

    if counts[label] >= 5:
        continue

    raw = row["image_data"]

    if raw.startswith("b'") or raw.startswith('b"'):
        image_bytes = ast.literal_eval(raw)
    else:
        image_bytes = base64.b64decode(raw)

    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    folder = "real" if label == 0 else "fake"
    path = f"dataset/quality_check/{folder}/sample_{counts[label] + 1}.png"

    img.save(path)

    counts[label] += 1
    print("SAVED:", folder, counts[label], img.size)

    if counts[0] >= 5 and counts[1] >= 5:
        break

print("DONE:", counts)
