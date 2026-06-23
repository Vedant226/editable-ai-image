"""
Standalone ComfyUI bridge app — Phase 1.

A separate FastAPI application (own port, default 8189) that mounts the ComfyUI
communication router. It exists so the bridge can be run and tested in complete
isolation: the existing inpaint service in app.py is never imported or modified.

Run:  ./run_comfy_bridge.sh        (from backend/)   — serves on 127.0.0.1:8189
or:   ../venv/bin/uvicorn comfy_app:app --host 127.0.0.1 --port 8189
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from comfy_router import router as comfy_router

app = FastAPI(title="Editable AI — ComfyUI Bridge")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(comfy_router)


@app.get("/health")
def health():
    """Liveness of the bridge process itself (not of ComfyUI)."""
    return {"service": "comfyui-bridge", "ok": True}
