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
- `followup` (optional but recommended): questions when still uncertain
- `meta` (optional): schema/model versions, latency

#### 4.3.1 `data.final` (UI-critical contract)
```json
{
  "category": "PET Plastic Bottle",
  "category_id": "plastic.pet_bottle",
  "recyclable": true,
  "confidence": 0.93,
  "instruction": "Rinse and empty containers.",
  "instructions": [
    "Empty all contents.",
    "Rinse to remove food residue.",
    "If heavily contaminated, dispose as general waste."
  ],
  "disposal_method": "Blue Recycling Bin",
  "bin_type": "blue",
  "rationale_tags": ["looks_like_bottle"]
}
```

Field semantics:
- `category` (Critical): short, user-facing item name (Tier-2 may refine beyond Tier-1 labels)
- `category_id` (Recommended): machine-readable taxonomy id (forward-compatible)
- `recyclable` (Critical): boolean used by app's primary UI logic
- `confidence` (Critical): system confidence in [0, 1]
- `instruction` (Critical): one-line instruction shown on the result card
- `instructions` (Recommended): 2-6 short imperative steps
- `disposal_method` (Optional): user-facing disposal channel text
- `bin_type` (Optional): normalized enum `blue|general|ewaste|textile|special|unknown`
- `rationale_tags` (Optional): <=3 tags for debugging/explainability

#### 4.3.2 `data.decision` (Tier-2 usage transparency)
```json
{
  "used_tier2": false,
  "reason_codes": ["LOW_CONFIDENCE", "LOW_MARGIN"],
  "thresholds": {
    "conf_threshold": 0.70,
    "margin_threshold": 0.12
  }
}
```

#### 4.3.3 `data.followup` (uncertainty handling)
```json
{
  "needs_confirmation": true,
  "questions": [
    {
      "id": "q1",
      "type": "single_choice",
      "question": "Is there food residue?",
      "options": ["Yes","No","Not sure"]
    }
  ]
}
```

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
  "model_versions": {"tier1":"onnx_v1.0.0","tier2":"llm_v0.1.0"},
  "latency_ms": {"total": 2650, "tier2": 1900}
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
- `decision.used_tier2=false`
- `final.category="Uncertain"`
- `followup.needs_confirmation=true` and 2-3 targeted questions

---

## 6) Tier-2 Expert Requirements (Future)

Tier-2 output must satisfy:
- category (short user-facing)
- category_id (recommended)
- recyclable boolean
- instruction + instructions (actionable disposal steps)
- optional: bin_type/disposal_method

If Tier-2 remains uncertain:
- return followup questions (do NOT guess confidently)

Tier-2 adapter must:
- enforce JSON-only outputs
- validate strict schema
- repair/fallback to "uncertain + followup" on invalid outputs

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
- used_tier2, tier2_latency
- final.bin_type, final.category_id
- error_code (if any)

Recommended metrics:
- Tier-2 invocation rate, timeout rate
- end-to-end latency percentiles
- distribution of final categories
- "uncertain + followup" rate

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
  val category_id: String? = null,
  val recyclable: Boolean,
  val confidence: Double,
  val instruction: String,
  val instructions: List<String> = emptyList(),
  val disposal_method: String? = null,
  val bin_type: String? = null,
  val rationale_tags: List<String>? = null
)

data class FollowupQuestion(
  val id: String,
  val type: String,
  val question: String,
  val options: List<String>
)

data class Followup(
  val needs_confirmation: Boolean,
  val questions: List<FollowupQuestion> = emptyList()
)

data class ScanData(
  val tier1: Tier1Result? = null,
  val decision: Decision,
  val final: FinalResult,
  val followup: Followup? = null,
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

---

## 12) Reference Python Implementation (FastAPI)

Repo reference implementation:
- `CNN/services/scan_service_v0_1.py`

Run locally (from repo root):
```bash
uvicorn CNN.services.scan_service_v0_1:app --host 0.0.0.0 --port 8000
```

Health check:
```bash
curl http://127.0.0.1:8000/health
```
