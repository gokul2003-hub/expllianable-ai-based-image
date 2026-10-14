import requests
import base64
import ast
import os
from PIL import Image
from io import BytesIO

url = "https://datasets-server.huggingface.co/rows?dataset=OwensLab%2FCommunityForensics-Small&config=default&split=train&offset=0&length=10"

data = requests.get(url, timeout=60).json()

os.makedirs("dataset/quality_check", exist_ok=True)

for r in data["rows"]:
    row = r["row"]
    raw = row["image_data"]

    if raw.startswith("b'") or raw.startswith('b"'):
        image_bytes = ast.literal_eval(raw)
    else:
        image_bytes = base64.b64decode(raw)

    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    filename = f"dataset/quality_check/{r['row_idx']}_{row['label']}.png"
    img.save(filename)

    print("SAVED:", filename, img.size, row["mode"])

print("DONE")
