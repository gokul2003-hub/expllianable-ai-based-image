import pyarrow.parquet as pq
import requests
import os

url = "https://huggingface.co/datasets/OwensLab/CommunityForensics-Small/resolve/main/data/HFCF_small_0.parquet"

r = requests.get(url, headers={"Range": "bytes=0-65535"}, timeout=30)
print("STATUS:", r.status_code)
print("BYTES:", len(r.content))
print("DONE")
