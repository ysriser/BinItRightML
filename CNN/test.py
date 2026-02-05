import json
import requests

url = "http://127.0.0.1:8000/api/v1/scan"
img_path = r"H:\GDipSA61_ADProject_Repo\BinItRightML\CNN\data\G3_SGData\glass\glass 2449.jpg"

tier1 = {
  "category": "other_uncertain",
  "confidence": 0.91,
  "top3": [
    {"label":"plastic","p":0.91},
    {"label":"glass","p":0.05},
    {"label":"other_uncertain","p":0.02}
  ],
  "escalate": False
}

with open(img_path, "rb") as f:
    files = {"image": ("test.jpg", f, "image/jpeg")}
    data = {
        "tier1": json.dumps(tier1),
        "timestamp": "1730000000000"
    }
    r = requests.post(url, files=files, data=data, timeout=30)
    print(r.status_code)
    print(r.text)
