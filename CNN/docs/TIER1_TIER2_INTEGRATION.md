# Tier-1 / Tier-2 Integration (Android + Spring Boot + Python)

Audience: Android developers, Spring Boot/backend developers, and ML team members.

Goal: Integrate our **Tier-1** on-device classifier (ONNX) into the product flow, and define a clean interface for a future **Tier-2 Expert** model that handles uncertain/complex items.

Repo: `BinItRightML` (module under `CNN/`).

For the stable Android/Spring Boot API contract, see:
- `CNN/docs/SCAN_SERVICE_SPEC_v0_1.md`

---

## 0) TL;DR (What You Need To Know)

### Tier-1 (fast, cheap)
- Runs on Android via ONNX Runtime (~200ms on device).
- Predicts one of 7 labels:
  - `paper`, `plastic`, `metal`, `glass`, `e-waste`, `textile`, `other_uncertain`
- Produces: `top1`, `top3`, and an `escalate` boolean.

### Tier-2 (slow, expensive, higher accuracy on tricky items)
- Only called when Tier-1 is uncertain (`escalate=true`) or returns `other_uncertain`.
- Runs on server (Spring Boot calls a Python service or a cloud model).
- Returns refined category + actionable disposal instructions (no quiz/follow-up in v0.1).

### Source of truth for preprocessing and thresholds
- Model files (Tier-1):
  - `CNN/models/tier1.onnx`
  - `CNN/models/label_map.json`
  - `CNN/models/infer_config.json`
- Tier-1 reject + multicrop config:
  - `CNN/experiments/v1_multicrop_reject/configs/infer_v1.yaml`
- Shared preprocessing/decision code:
  - `CNN/shared/preprocess.py`
  - `CNN/shared/decision.py`
  - `CNN/shared/multicrop.py`

---

## 1) System Overview (End-to-End Flow)

### Recommended product flow
1) Android captures a photo (usually 4:3 / 16:9), UI preview is 1:1 and user is encouraged to center the object.
2) Android runs Tier-1 ONNX locally:
   - Outputs `category`, `confidence`, `top3`, and `escalate`.
3) If `escalate=false`:
   - Show Tier-1 result + instructions.
4) If `escalate=true`:
   - Show "Analyzing..." UI.
   - Send the photo + Tier-1 top3 to backend.
   - Backend calls Tier-2 expert and returns final result + instructions.

### Why we do this
We optimize for "selective reliability":
- When Tier-1 is confident (not escalated), it should be very accurate.
- When Tier-1 is not confident, we prefer to say "uncertain" and escalate.

---

## 2) Tier-1 Model Contract (Android ORT)

### Files shipped to Android
- `tier1.onnx` (model)
- `label_map.json` (index -> label)
- `infer_config.json` (preprocess params + basic thresholds)

### Model input tensor
- Shape: `[1, 3, img_size, img_size]`
- DType: `float32`
- Channel order: `RGB`
- Pixel scaling: `0..255` -> `0..1` -> normalize

### Model output
- `logits`: shape `[1, num_classes]` float32
- Convert to probabilities with softmax:
  - `probs = softmax(logits)`

### Labels (final outputs)
We keep this fixed for Tier-1:
```
paper
plastic
metal
glass
other_uncertain
e-waste
textile
```

The label order MUST match `CNN/models/label_map.json`.

---

## 3) Tier-1 Preprocessing Spec (Must Match Android + Python)

### Why preprocessing is critical
If Android preprocessing differs from training/eval preprocessing, accuracy will drop sharply.

### Current Tier-1 preprocess (center crop pipeline)
Source code: `CNN/shared/preprocess.py`

Definition (`mode="center"`):
1) Resize the image so that the **shorter side** becomes `round(img_size * resize_scale)`.
2) Take a **center crop** of size `img_size x img_size`.
3) Convert to float tensor, normalize by mean/std.

This performs a **square crop without stretching**:
- It effectively keeps `1 / resize_scale` of the short side.
- Example: `resize_scale=1.15` => keep ~`0.87` of the short side (center region).

Where values come from:
- `img_size`, `mean`, `std` are read from `CNN/models/infer_config.json`.
- Multi-crop `resize_scale` comes from `infer_v1.yaml` under `multicrop.resize_scale`.

### Supported formats
Python side supports: JPG/PNG/WEBP/TIFF and HEIC/HEIF (via `pillow-heif`).
Android should upload JPEG/PNG to simplify cross-platform behavior.

---

## 4) Tier-1 Confidence / Reject / Escalation Logic

Source code: `CNN/shared/decision.py`
Config: `CNN/experiments/v1_multicrop_reject/configs/infer_v1.yaml`

### Metrics
- `max_prob`: top1 probability
- `margin`: `top1_prob - top2_prob`
- `entropy`: `-sum(p * log(p))` (higher = more uncertain)

### Escalation rules (conceptual)
We set `escalate=true` if ANY condition triggers:
- predicted label is `other_uncertain`
- `max_prob < conf_threshold`
- `margin < margin_threshold`
- optional: `entropy > entropy_threshold`
- optional: stricter per-class thresholds (e.g., plastic/glass)

If `reject_to_other=true` and escalated:
- `final_label` is forced to `other_uncertain` (safe default).

This is what enables "selective accuracy".

---

## 5) Multi-crop Inference (Tier-1 v1)

Purpose: If a photo is borderline, try extra crops cheaply to improve confidence.

Code: `CNN/shared/multicrop.py`
Config: `infer_v1.yaml` under `multicrop:`

### How it works
1) Run a normal "center" crop.
2) If not confident (trigger thresholds), run extra crops:
   - `zoom_center` (center zoom then crop)
   - `full_resize` (resize to square; no crop)
3) Combine crop results by:
   - `avg` (average probs) OR
   - `best_score` (pick crop with best `max_prob + margin`)

Multi-crop should only run when needed (to keep runtime reasonable).

---

## 6) Tier-1 Python Inference API (Optional, for QA/Backend)

### 6.1 Stable Scan Contract (v0.1)

For Android/Spring Boot integration, the stable contract is:
- `CNN/docs/SCAN_SERVICE_SPEC_v0_1.md`

Reference Python implementation (v0.1 envelope + rules mapping, Tier-2 not yet wired):
- `CNN/services/scan_service_v0_1.py`

Run:
```bash
uvicorn CNN.services.scan_service_v0_1:app --host 0.0.0.0 --port 8000
```

### 6.2 Tier-1 Debug/QA Server (legacy payload)

We also provide a FastAPI wrapper for Tier-1 v1 (multi-crop + reject):

File:
- `CNN/experiments/v1_multicrop_reject/serve_fastapi_v1.py`

### Endpoints
- `GET /health` -> `{"status":"ok"}`
- `POST /api/v1/scan` (multipart/form-data)
  - Field name: `image`

### Response JSON (exact)
```json
{
  "category": "plastic",
  "confidence": 0.91,
  "top3": [
    {"label": "plastic", "p": 0.91},
    {"label": "glass", "p": 0.05},
    {"label": "other_uncertain", "p": 0.02}
  ],
  "escalate": false,
  "instructions": ["Rinse and empty containers.", "Remove caps if possible."],
  "expert": null,
  "debug": {
    "reasons": [],
    "max_prob": 0.91,
    "margin": 0.86,
    "entropy": 0.31,
    "multicrop_used": false,
    "selected_crop": "center"
  }
}
```

This API is primarily for:
- validating model behavior on laptops/servers
- giving Spring Boot an easy way to call Tier-1 if needed

Production recommendation remains: Tier-1 runs on Android.

---

## 7) Tier-2 Expert Interface (Proposed)

Tier-2 is not implemented yet. This section defines a clean contract so Android/Spring Boot can integrate early.

### When to call Tier-2
Call Tier-2 if Tier-1 returns:
- `escalate=true` OR
- `category == "other_uncertain"`

### Recommended Tier-2 request (Android -> Spring Boot -> Tier-2 service)
- image (bytes, JPEG recommended)
- tier1 top3 list (labels + probabilities)
- optional: device hints / user hints

Example JSON + image (multipart):
- Field: `image` (file)
- Field: `tier1_top3` (JSON string)

Example `tier1_top3`:
```json
[
  {"label":"other_uncertain","p":0.41},
  {"label":"plastic","p":0.33},
  {"label":"paper","p":0.12}
]
```

### Recommended Tier-2 response (server -> app)
Tier-2 should return:
- final category (may be one of Tier-1 labels or a more detailed subtype)
- `instruction` + `instructions` (step-by-step dismantling/disposal)
- `recyclable` boolean (blue-bin flow)
- `confidence` (0-1)

v0.1 policy:
- Do NOT return quiz/follow-up questions in the response body.
- If uncertain, return `category="Uncertain"` and provide conservative disposal steps (general waste) + safety note (battery/e-waste).

Example (Tier-2 output fields, not the full scan envelope):
```json
{
  "category": "PET Plastic Bottle",
  "recyclable": true,
  "confidence": 0.93,
  "instruction": "Empty and rinse before recycling.",
  "instructions": [
    "Empty all contents.",
    "Rinse to remove residue.",
    "Remove caps if possible."
  ]
}
```

### Backend ownership
- Spring Boot owns routing, auth, rate limits, caching, and analytics logging.
- Tier-2 service can be Python (FastAPI) or a managed cloud model endpoint.

---

## 8) Parity & Debugging (Android vs Python)

We have a permanent parity self-test tool:
- `CNN/experiments/v1_parity_self_test/`

It can:
- generate a golden snapshot of outputs
- compare future changes (detect preprocessing drift)
- compare Android-exported tensor/probs vs Python inference

This is critical when "accuracy suddenly drops".

---

## 9) Model Updates / Versioning

Model promotion convention (current):
- canonical model files live in `CNN/models/`
  - `tier1.onnx`
  - `tier1_best.pt`
  - `infer_config.json`
  - `label_map.json`

Training scripts (v2 robust fine-tune) export into:
- `CNN/experiments/v2_robust_finetune/artifacts/<run_name>/`
and then update `CNN/models/` automatically.

---

## Appendix A: Useful Commands

### Run Tier-1 server (for QA)
```powershell
python CNN/experiments/v1_multicrop_reject/serve_fastapi_v1.py
```

### Evaluate thresholds on SG data
```powershell
python CNN/experiments/v1_multicrop_reject/scripts/run_eval_custom_v1.py
```

### Train v2 robust fine-tune
```powershell
python CNN/experiments/v2_robust_finetune/scripts/train_v2.py ^
  --config CNN/experiments/v2_robust_finetune/configs/train_v2.yaml
```
