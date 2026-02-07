"""Quick local client to test the Scan API response.

Edit the CONFIG section below, then run from repo root:
  .venv\\Scripts\\python.exe CNN\\tools\\scan_api_client.py

This script is intentionally simple so Android/backend teammates can reproduce
API behavior without learning curl or writing code.

Endpoint contract:
  - POST /api/v1/scan (multipart/form-data)
  - fields: image (required), tier1 (optional JSON string), timestamp (optional), force_cloud (optional)

Server entrypoint (in another terminal):
  python -m uvicorn CNN.services.scan_service_v0_1:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import mimetypes
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx


# =========================
# CONFIG (EDIT ME)
# =========================
BASE_URL = "http://127.0.0.1:8000"
# IMAGE_PATH = r"H:\GDipSA61_ADProject_Repo\BinItRightML\CNN\data\G3_SGData\metal\metal_001.jpg"
# IMAGE_PATH = r"H:\GDipSA61_ADProject_Repo\BinItRightML\CNN\data\G3_SGData\other_uncertain\cardboard 421.jpg"
IMAGE_PATH = r"H:\GDipSA61_ADProject_Repo\BinItRightML\CNN\data\G3_SGData\other_uncertain\cardboard 618.jpg"
FORCE_CLOUD = True
TIMESTAMP_MS: Optional[int] = None

# If you set TIER1_JSON=None, the server may run Tier-1 on-server (fallback).
# If you set it to a dict, the server will use your Tier-1 result.
TIER1_JSON: Optional[dict[str, Any]] = None
# Example:
# TIER1_JSON = {
#     "category": "plastic",
#     "confidence": 0.91,
#     "top3": [
#         {"label": "plastic", "p": 0.91},
#         {"label": "glass", "p": 0.05},
#         {"label": "other_uncertain", "p": 0.02},
#     ],
#     "escalate": False,
# }

# Increase if your network is slow / you're using Tier-2.
TIMEOUT_S = 30.0


def _guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def main() -> int:
    image_path = Path(IMAGE_PATH)
    if not image_path.exists():
        print(f"[error] Image not found: {image_path}", file=sys.stderr)
        return 2

    url = f"{BASE_URL.rstrip('/')}/api/v1/scan"
    data: dict[str, str] = {}
    if FORCE_CLOUD:
        data["force_cloud"] = "true"
    if TIMESTAMP_MS is not None:
        data["timestamp"] = str(int(TIMESTAMP_MS))
    if TIER1_JSON is not None:
        data["tier1"] = json.dumps(TIER1_JSON, ensure_ascii=False)

    mime = _guess_mime_type(image_path)
    t0 = time.perf_counter()
    try:
        with image_path.open("rb") as f:
            files = {"image": (image_path.name, f, mime)}
            resp = httpx.post(url, data=data, files=files, timeout=TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 - local debug script
        print(f"[error] Request failed: {exc}", file=sys.stderr)
        return 3

    dt_ms = int(round((time.perf_counter() - t0) * 1000.0))
    print(f"HTTP {resp.status_code} ({dt_ms} ms) -> {url}")

    content_type = resp.headers.get("content-type", "")
    if "application/json" in content_type:
        body = resp.json()
        print(json.dumps(body, indent=2, ensure_ascii=False))

        # Quick extraction for UI-critical fields.
        data_obj = body.get("data") or {}
        final = data_obj.get("final") or {}
        meta = data_obj.get("meta") or {}
        print("\n[final]")
        print(json.dumps(final, indent=2, ensure_ascii=False))

        print("\n[meta]")
        for k in ("tier2_provider_attempted", "tier2_provider_used", "tier2_provider", "tier2_error"):
            if k in meta:
                print(f"- {k}: {meta[k]}")

        return 0

    print(resp.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
