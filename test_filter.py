import requests

BASE = "https://datasets-server.huggingface.co/filter"

for label in [0, 1]:
    params = {
        "dataset": "OwensLab/CommunityForensics-Small",
        "config": "default",
        "split": "train",
        "where": f'"label"={label}',
        "offset": 0,
        "length": 5,
    }

    r = requests.get(BASE, params=params, timeout=60)
    print("LABEL:", label)
    print("STATUS:", r.status_code)

    data = r.json()
    print("ROWS:", len(data.get("rows", [])))

    for item in data.get("rows", []):
        row = item["row"]
        print(row["label"], row["resolution"], row["mode"])
