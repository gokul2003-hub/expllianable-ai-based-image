import requests
import base64
import ast
import os
from PIL import Image
from io import BytesIO

url = "https://datasets-server.huggingface.co/rows?dataset=OwensLab%2FCommunityForensics-Small&config=default&split=train&offset=10000&length=5"

data = requests.get(url, timeout=60).json()

os.makedirs("dataset/quality_check/real", exist_ok=True)

for i, item in enumerate(data["rows"], 1):
    raw = item["row"]["image_data"]

    if raw.startswith("b'") or raw.startswith('b"'):
        image_bytes = ast.literal_eval(raw)
    else:
        image_bytes = base64.b64decode(raw)

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    path = f"dataset/quality_check/real/sample_{i}.png"
    img.save(path)

    print("SAVED:", path, img.size)

print("DONE")
