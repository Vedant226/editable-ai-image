#!/usr/bin/env bash
# Start the ComfyUI bridge on 127.0.0.1:8189 using the project venv.
# Isolated from the inpaint service (run.sh / app.py on :8000).
set -euo pipefail
cd "$(dirname "$0")"
exec ../venv/bin/uvicorn comfy_app:app --host 127.0.0.1 --port 8189 "$@"
