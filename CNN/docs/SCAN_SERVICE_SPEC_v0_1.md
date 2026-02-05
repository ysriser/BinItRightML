# BinItRight Scan Service (Tier-1 + Tier-2) Engineering Specification

Version: v0.1 (Contract Baseline)  
Audience: Android Client, Backend/Spring Boot, ML (On-device + Server), QA/DevOps

This document defines the **stable integration contract** for the "Scan Item" feature using a two-tier strategy:
- **Tier-1 (on-device, Android)**: fast, low-cost, ONNX Runtime. Produces Tier-1 labels + confidence signals.
- **Tier-2 (server expert)**: slower, higher cost. Called only when Tier-1 is uncertain. Produces refined category + actionable instructions.

This spec is what we align on across teams before implementing Tier-2.

---

## 1) Scope and Goals

### Goals (v0.1)
Primary outcomes required by the mobile UI:
1) User-facing category name (short)
2) `recyclable` boolean
3) Confidence score (0-1) for "scan again" prompts
4) One prominent instruction + optional step-by-step instructions
5) Optional disposal method / bin type for clearer UX

### Non-goals (v0.1)
1) Location-based disposal point lookup
2) Personalization (user profiles, habits)
3) Multi-item detection (more than one object per image)
4) Long-running chat dialogue (Tier-2 is an "expert decision", not a chatbot)

---

## 2) System Overview and Architecture

### 2.1 External view (Android-facing, stable)
Android sends one request to a single endpoint:
- `POST /api/v1/scan` (multipart/form-data)

Server returns one JSON response containing **UI-ready fields** regardless of whether Tier-2 was invoked.

### 2.2 Internal logical modules (implementation may vary)
Reference decomposition:
1) Scan API (Gateway)
2) Tier-1 Resolver
   - If Android provides Tier-1 results: use them
   - Else: optionally run Tier-1 on server (parity/QA/fallback)
3) Decision Engine (escalation rules)
4) Tier-2 Expert Adapter (optional path)
5) Instruction Composer
6) Response Assembler

### 2.3 Deployment topology options
The external contract is identical for these:
- Option A: Android -> Python FastAPI (Tier-1 optional fallback + Tier-2 adapter)
- Option B: Android -> Spring Boot (auth/limits/logging) -> Python service (Tier-1/Tier-2)
- Option C: Android -> Spring Boot (Tier-2 only). Tier-1 strictly on-device.

---

## 3) Tier-1 Model Contract (Source-of-Truth Alignment)

### 3.1 Fixed Tier-1 labels
Tier-1 outputs one of these fixed labels:
```
paper
plastic
metal
glass
e-waste
textile
other_uncertain
```

Source of truth:
- `CNN/models/label_map.json` (exact spelling and ordering)

### 3.2 Tier-1 output semantics
Tier-1 produces:
- `category`: top-1 label from the fixed set
- `confidence`: top-1 probability in [0, 1]
- `top3`: ordered list of top-3 labels + probabilities
- `escalate`: boolean indicating uncertainty/complexity (should call Tier-2)

Tier-1 may also compute debug metrics:
- `margin = top1_prob - top2_prob`
- `entropy = -sum(p*log(p))`
- `multicrop_used`, `selected_crop`

---

## 4) Public API Specification (Android-facing)

### 4.1 Endpoint
`POST /api/v1/scan`  
`Content-Type: multipart/form-data`

### 4.2 Request fields
`image` (required)
- Type: binary file
- Formats: JPEG or PNG (JPEG recommended)
- Recommended limits: max 8 MB (413 on violation)

`timestamp` (optional)
- Type: Long / integer
- Unit: Unix milliseconds

`tier1` (optional)
- Type: string (JSON)
- Purpose: pass on-device Tier-1 results to server to skip duplicate inference and help Tier-2 reasoning

Recommended `tier1` JSON schema:
```json
{
  "category": "plastic",
  "confidence": 0.91,
  "top3": [
    {"label":"plastic","p":0.91},
    {"label":"glass","p":0.05},
    {"label":"other_uncertain","p":0.02}
  ],
  "escalate": false
}
```

`force_cloud` (optional)
- Type: string (`"true"` / `"false"`)
- Default: `"false"`
- Purpose: **debug trigger** to force the Tier-2 path even when Tier-1 is confident.
  This is useful to test the end-to-end "upgrade path" before a real Tier-2 LLM/VLM is integrated.

### 4.3 Success response (HTTP 200)
Top-level envelope:
```json
{
  "status": "success",
  "request_id": "uuid-string",
  "data": { }
}
```

`data` fields:
- `tier1` (optional, object or null): echoes Tier-1 results (from Android or server fallback)
- `decision` (required): Tier-2 usage and reason codes
- `final` (required): UI-ready final decision and instructions
- `meta` (optional): schema/model versions, latency

#### 4.3.1 `data.final` (UI-critical contract)
Android UI MUST only depend on the 5 critical fields:
`category`, `recyclable`, `confidence`, `instruction`, `instructions`.
Other fields are considered **legacy** (kept temporarily to avoid breaking existing clients).

```json
{
  "category": "PET Plastic Bottle",
  "recyclable": true,
  "confidence": 0.93,
  "instruction": "Rinse and empty containers.",
  "instructions": [
    "Empty all contents.",
    "Rinse to remove food residue.",
    "If heavily contaminated, dispose as general waste."
  ]
}
```

Field semantics:
- `category` (Critical): short, user-facing item name (Tier-2 may refine beyond Tier-1 labels)
  - Special routing convention: if it is a special stream, the category MUST start with:
    - `E-waste - ...`
    - `Textile - ...`
- `recyclable` (Critical): **blue-bin flow** boolean (true ONLY if normal recycling bin flow)
  - E-waste/Textile should be `false` here (separate streams, not blue bin)
- `confidence` (Critical): system confidence in [0, 1]
- `instruction` (Critical): one-line instruction shown on the result card
- `instructions` (Critical): 2-8 short imperative disposal steps (no scanning tips)

Not in v0.1 contract:
- `followup`, `questions` (quiz is a separate UI flow)
- `bin_type`, `disposal_method`, `category_id` (Android routes by `final.recyclable` + keywords/prefix in `final.category`)

#### 4.3.2 `data.decision` (Tier-2 usage transparency)
```json
{
  "used_tier2": false,
  "reason_codes": ["LOW_CONFIDENCE", "LOW_MARGIN"],
  "thresholds": {
    "conf_threshold": 0.70,
    "margin_threshold": 0.05,
    "conf_threshold_plastic": 0.80,
    "conf_threshold_glass": 0.80
  }
}
```

#### 4.3.3 No followup/quiz in v0.1
To keep Android integration simple and avoid UI churn, v0.1 does **not** return:
- `data.followup`
- any `questions` / quiz fields anywhere in the response body (including `meta`)

If the system is uncertain, it should still return HTTP 200 with a valid `data.final`
where `category="Uncertain"` and `instructions` provide conservative disposal guidance
(default: general waste) plus safety notes (battery/electronics => e-waste).

Instruction content policy (v0.1):
- `instruction` / `instructions` MUST be disposal steps only.
- Do NOT include camera/scanning tips (lighting/framing/rescan prompts) in these fields.

#### 4.3.4 `data.tier1` (echoed for parity/debug)
```json
{
  "category": "other_uncertain",
  "confidence": 0.41,
  "top3": [
    {"label":"other_uncertain","p":0.41},
    {"label":"plastic","p":0.33},
    {"label":"paper","p":0.12}
  ],
  "escalate": true
}
```

#### 4.3.5 `data.meta` (optional)
```json
{
  "schema_version": "0.1",
  "latency_ms": {"total": 2650},
  "tier2_provider_attempted": "openai",
  "tier2_provider_used": "openai",
  "tier2_provider": "openai"
}
```

---

## 5) Error Handling Contract

### 5.1 Error response (non-200)
```json
{
  "status": "error",
  "code": "INVALID_IMAGE",
  "message": "User-friendly error message."
}
```

Recommended HTTP status codes:
- 400: `INVALID_IMAGE`, `MISSING_IMAGE`
- 413: `IMAGE_TOO_LARGE`
- 415: `UNSUPPORTED_MEDIA_TYPE`
- 429: `RATE_LIMITED`
- 503: `TIER2_UNAVAILABLE`
- 504: `TIER2_TIMEOUT`
- 500: `INTERNAL_ERROR`

### 5.2 Graceful degradation (Tier-2 failure)
If Tier-2 times out/unavailable, prefer returning HTTP 200 with:
- `final.category="Uncertain"` (and valid critical fields)
- no followup/questions (v0.1 rule)

---

## 6) Tier-2 Expert Requirements (Future)

Tier-2 output must satisfy:
- category (short user-facing)
- category_id (recommended)
- recyclable boolean
- instruction + instructions (actionable disposal steps)
- optional: bin_type/disposal_method

Tier-2 adapter must:
- enforce JSON-only outputs
- validate strict schema
- repair/fallback to a safe `final` (e.g., `category="Uncertain"`) on invalid outputs

---

## 7) Security and Secrets

Client-side:
- Do not embed Tier-2 API keys in APK.

Server-side:
- Store keys in env/secret manager.

Protection (recommended):
- rate limits
- request size limits
- optional caching by image hash
- structured logging with `request_id`

---

## 8) Observability and QA

Recommended log fields:
- request_id, timestamp, image_size
- tier1.category, tier1.confidence, tier1.escalate
- used_tier2, meta.tier2_provider_used, meta.tier2_error.code/http_status (if any)
- final.category, final.recyclable, final.confidence
- error_code (if any)

Recommended metrics:
- Tier-2 invocation rate, timeout rate
- end-to-end latency percentiles
- distribution of final categories
- uncertainty rate (e.g., `final.category` starts with "Uncertain")

Parity testing:
- `CNN/experiments/v1_parity_self_test/`

---

## 9) Implementation Status (What Exists In This Repo)

### Tier-1 artifacts (canonical)
- `CNN/models/tier1.onnx`
- `CNN/models/label_map.json`
- `CNN/models/infer_config.json`

### Tier-1 reject + multicrop logic
- `CNN/experiments/v1_multicrop_reject/configs/infer_v1.yaml`
- `CNN/shared/decision.py`
- `CNN/shared/multicrop.py`
- `CNN/shared/preprocess.py`

### Existing Python FastAPI servers (legacy response)
These return a simpler JSON (category/confidence/top3/escalate/instructions):
- `CNN/clf_7cats_tier1/serve.py`
- `CNN/experiments/v1_multicrop_reject/serve_fastapi_v1.py`

They are useful for QA, but the **external contract** for Android/Spring Boot is this document.

### Do we need to retrain to adopt this contract?
No. This is primarily a **service contract**. Retraining is only needed if we change:
- label set / label order
- preprocessing (img_size/mean/std/resize_scale/crop logic)

---

## 10) Kotlin DTOs (Android)

These are examples of client-side models to parse/compose JSON.

```kotlin
data class TopK(val label: String, val p: Double)

data class Tier1Result(
  val category: String,
  val confidence: Double,
  val top3: List<TopK>,
  val escalate: Boolean
)

data class Decision(
  val used_tier2: Boolean,
  val reason_codes: List<String> = emptyList(),
  val thresholds: Map<String, Double>? = null
)

data class FinalResult(
  val category: String,
  val recyclable: Boolean,
  val confidence: Double,
  val instruction: String,
  val instructions: List<String> = emptyList()
)

data class ScanData(
  val tier1: Tier1Result? = null,
  val decision: Decision,
  val final: FinalResult,
  val meta: Map<String, Any>? = null
)

data class ScanResponse(
  val status: String,
  val request_id: String,
  val data: ScanData? = null,
  val code: String? = null,
  val message: String? = null
)
```

---

## 11) Example curl (multipart with tier1)

```bash
curl -X POST https://<host>/api/v1/scan \
  -F "image=@test.jpg" \
  -F 'tier1={"category":"plastic","confidence":0.91,"top3":[{"label":"plastic","p":0.91},{"label":"glass","p":0.05},{"label":"other_uncertain","p":0.02}],"escalate":false}' \
  -F "timestamp=1730000000000"
```

Force Tier-2 path (debug):
```bash
curl -X POST https://<host>/api/v1/scan \
  -F "image=@test.jpg" \
  -F 'tier1={"category":"plastic","confidence":0.91,"top3":[{"label":"plastic","p":0.91},{"label":"glass","p":0.05},{"label":"other_uncertain","p":0.02}],"escalate":false}' \
  -F "force_cloud=true"
```

PowerShell example (use OpenAI Tier-2, then force Tier-2 via request):
```powershell
$env:TIER2_PROVIDER="openai"
$env:OPENAI_API_KEY="YOUR_KEY"
# or: $env:LLM_API_KEY="YOUR_KEY"

curl.exe -X POST http://127.0.0.1:8000/api/v1/scan `
  -F "image=@test.jpg" `
  -F 'tier1={\"category\":\"plastic\",\"confidence\":0.91,\"top3\":[{\"label\":\"plastic\",\"p\":0.91},{\"label\":\"glass\",\"p\":0.05},{\"label\":\"other_uncertain\",\"p\":0.02}],\"escalate\":false}' `
  -F "force_cloud=true"
```

---

## 12) Reference Python Implementation (FastAPI)

Repo reference implementation:
- `CNN/services/scan_service_v0_1.py`

Run locally (from repo root):
```bash
uvicorn CNN.services.scan_service_v0_1:app --host 0.0.0.0 --port 8000
```

Tier-2 provider switch (server-side env vars):
- `TIER2_PROVIDER=mock|openai` (default: `mock`)
- `OPENAI_API_KEY` (required when `TIER2_PROVIDER=openai`)
  - For local debugging we also accept `LLM_API_KEY` as an alias.
- `OPENAI_TIER2_MODEL` (optional, default: `gpt-5-mini`)
- `OPENAI_REASONING_EFFORT` (optional, default: `none`) - lower latency on GPT-5
- `OPENAI_VERBOSITY` (optional, default: `low`) - shorter outputs, lower latency

Windows examples:
```powershell
$env:TIER2_PROVIDER = "openai"
$env:OPENAI_API_KEY = "<your_key>"
# optional
$env:OPENAI_TIER2_MODEL = "gpt-5-mini"
uvicorn CNN.services.scan_service_v0_1:app --host 0.0.0.0 --port 8000
```

```bat
set TIER2_PROVIDER=openai
set OPENAI_API_KEY=<your_key>
REM optional
set OPENAI_TIER2_MODEL=gpt-5-mini
uvicorn CNN.services.scan_service_v0_1:app --host 0.0.0.0 --port 8000
```

Provider behavior:
- `openai`: Tier-2 generates `data.final` directly (strict JSON schema, no followup/questions).
- `mock`: Tier-2 uses deterministic templates.
- If OpenAI fails/timeout/invalid JSON: automatic fallback to `mock` and append
  `TIER2_FALLBACK_MOCK` to `data.decision.reason_codes`. `data.meta.tier2_provider`
  indicates the provider actually used (`openai` or `mock`).
  - For transparency, the server also returns:
    - `data.meta.tier2_provider_attempted`
    - `data.meta.tier2_provider_used`
    - `data.meta.tier2_error` (compact, no secrets) when fallback happens:
      `{http_status, code, message}`

Health check:
```bash
curl http://127.0.0.1:8000/health
```

Python requests example (force Tier-2 path):
```python
import json
import requests

url = "http://127.0.0.1:8000/api/v1/scan"

tier1 = {
  "category": "plastic",
  "confidence": 0.91,
  "top3": [
    {"label": "plastic", "p": 0.91},
    {"label": "glass", "p": 0.05},
    {"label": "other_uncertain", "p": 0.02},
  ],
  "escalate": False,
}

with open("test.jpg", "rb") as f:
  files = {"image": ("test.jpg", f, "image/jpeg")}
  data = {"tier1": json.dumps(tier1), "force_cloud": "true"}
  r = requests.post(url, files=files, data=data, timeout=30)
  print(r.status_code, r.json())
```
