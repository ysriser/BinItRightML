"""FastAPI entrypoint for container/CI deployments.

We expose the v0.1 Scan Service contract by default so Android/Spring Boot can
hit `/api/v1/scan` on the deployed Python service.

Spec:
- `CNN/docs/SCAN_SERVICE_SPEC_v0_1.md`
Implementation:
- `CNN/services/scan_service_v0_1.py`
"""

from __future__ import annotations

from CNN.services.scan_service_v0_1 import app


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok"}


@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    # Fix for Alert 10021
    response.headers["X-Content-Type-Options"] = "nosniff"
    # Fix for Alert 90004
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    return response
