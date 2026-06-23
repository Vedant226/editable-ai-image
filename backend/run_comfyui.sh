#!/usr/bin/env bash
# Start the ComfyUI server on 127.0.0.1:8188 — the SDXL engine the bridge talks to.
#
# THIS IS THE MISSING LAUNCHER that caused "ComfyUI not reachable at
# http://127.0.0.1:8188". The three services and their starters are:
#   run.sh              -> lift / inpaint service   (:8000, project venv)
#   run_comfy_bridge.sh -> ComfyUI bridge           (:8189, project venv)
#   run_comfyui.sh      -> ComfyUI itself           (:8188, ComfyUI/comfy_env venv)  <-- this file
#
# ComfyUI uses its OWN dedicated venv (ComfyUI/comfy_env), not the backend venv.
set -euo pipefail
cd "$(dirname "$0")/../ComfyUI"

# --lowvram keeps SDXL + controlnets within ~6 GB so ComfyUI coexists with the
# lift service (LaMa on CUDA) on an 8 GB GPU. Override by passing your own flags,
# e.g. ./run_comfyui.sh --normalvram, or set COMFY_VRAM="--medvram".
VRAM_FLAG="${COMFY_VRAM:---lowvram}"
PORT="${COMFY_PORT:-8188}"

exec ./comfy_env/bin/python main.py --port "$PORT" $VRAM_FLAG "$@"
