"""MimicCoach Modal app.

This is the smoke-test scaffold for P0. Real endpoints (pose extraction,
embedding, Qdrant query) land in P5. For now, /healthz is enough to verify
the deploy pipeline.
"""
from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]>=0.115", "pydantic>=2.9")
)

app = modal.App("mimiccoach", image=image)


@app.function()
@modal.fastapi_endpoint(method="GET", label="healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "mimiccoach"}


if __name__ == "__main__":
    print("Run with: modal serve backend/app.py  (dev) or  modal deploy backend/app.py")
