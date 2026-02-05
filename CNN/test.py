import os
import json
import base64
import requests
from pathlib import Path

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("Missing OPENAI_API_KEY (or LLM_API_KEY) in environment variables.")

# Choose a model you want to test. If you keep getting 404, it is often 'model not found / no access'.
MODEL = os.getenv("OPENAI_TIER2_MODEL", "gpt-4o-mini")
reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "minimal")
verbosity = os.getenv("OPENAI_VERBOSITY", "low")

BASE_URL = "https://api.openai.com"
HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
}

def pretty(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))

def call_models_list():
    print("\n[1] Testing key via GET /v1/models ...")
    r = requests.get(f"{BASE_URL}/v1/models", headers={"Authorization": HEADERS["Authorization"]}, timeout=10)
    if r.status_code != 200:
        print(f"HTTP {r.status_code}")
        try:
            pretty(r.json())
        except Exception:
            print(r.text)
        return None
    data = r.json()
    # Print a few model IDs for quick inspection
    ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
    print(f"OK. Total models visible: {len(ids)}")
    print("Sample model ids:")
    for mid in ids[:20]:
        print(" -", mid)
    return set(ids)

def call_responses_text(model: str):
    print(f"\n[2] Testing Responses API with model='{model}' ...")
    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Reply with exactly: OK"}
                ],
            }
        ],
        # Keep it short & cheap
        "max_output_tokens": 16,
    }
    r = requests.post(f"{BASE_URL}/v1/responses", headers=HEADERS, data=json.dumps(payload), timeout=15)
    print(f"HTTP {r.status_code}")
    try:
        j = r.json()
    except Exception:
        print(r.text)
        return

    # If error, show full structured error
    if r.status_code != 200:
        pretty(j)
        return

    # Extract output text if present
    # Responses output can be nested; simplest is to print the whole response id + a best-effort text.
    print("response_id:", j.get("id"))
    text_out = None
    try:
        # Try common path: output[0].content[0].text
        out0 = j.get("output", [])[0]
        c0 = out0.get("content", [])[0]
        text_out = c0.get("text")
    except Exception:
        pass
    print("text_out:", text_out if text_out is not None else "(could not auto-extract; see full JSON below)")
    if text_out is None:
        pretty(j)

def call_responses_with_image(model: str, image_path: str):
    print(f"\n[3] Testing Responses API with image input model='{model}' ...")
    p = Path(image_path)
    if not p.exists():
        raise SystemExit(f"Image not found: {image_path}")

    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    # naive mime guess
    suffix = p.suffix.lower()
    mime = "image/jpeg" if suffix in [".jpg", ".jpeg"] else "image/png"

    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is this object? Reply in 1 short sentence."},
                    {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
                ],
            }
        ],
        "max_output_tokens": 64,
    }
    r = requests.post(f"{BASE_URL}/v1/responses", headers=HEADERS, data=json.dumps(payload), timeout=30)
    print(f"HTTP {r.status_code}")
    try:
        j = r.json()
    except Exception:
        print(r.text)
        return

    if r.status_code != 200:
        pretty(j)
        return

    print("response_id:", j.get("id"))
    # Best-effort extract
    text_out = None
    try:
        out0 = j.get("output", [])[0]
        c0 = out0.get("content", [])[0]
        text_out = c0.get("text")
    except Exception:
        pass
    print("text_out:", text_out if text_out is not None else "(could not auto-extract; see full JSON below)")
    if text_out is None:
        pretty(j)

if __name__ == "__main__":
    visible_models = call_models_list()

    # Text-only test (fast): verifies endpoint + model access
    if visible_models is not None and MODEL not in visible_models:
        print(f"\nWARNING: model '{MODEL}' is NOT in /v1/models list for this key.")
        print("This often explains HTTP 404 when calling it. Try setting OPENAI_TIER2_MODEL to one that appears in the list.")
    call_responses_text(MODEL)

    # Optional: image test (comment out if you only want text check)
    # Set env IMAGE_PATH or edit below
    img = os.getenv("IMAGE_PATH")
    if img:
        call_responses_with_image(MODEL, img)
    else:
        print("\n[3] Skipped image test. To run it, set env var IMAGE_PATH to a jpg/png path.")
        print(r'Example (PowerShell): $env:IMAGE_PATH="H:\path\to\test.jpg"')
