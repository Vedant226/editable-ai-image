#!/usr/bin/env bash
# install_ai_models.sh — download the optional models that unlock the Phase-3
# quality upgrades for the ComfyUI AI editing engine.
#
# Everything here is OPTIONAL: the engine auto-detects what's installed and falls
# back gracefully. After running, restart ComfyUI and check:
#     curl -s localhost:8189/comfyui/capabilities | python3 -m json.tool
#
# Tuned for an RTX 4060 (8 GB): downloads ONE ControlNet "union" model (covers
# canny/depth/soft-edge/pose) + ONE tile model. Use at most one ControlNet per
# render (the engine already does) and ComfyUI's --lowvram if you hit OOM.
#
# Usage:
#   ./install_ai_models.sh controlnet      # ControlNet union + tile (~5 GB)  [recommended]
#   ./install_ai_models.sh ipadapter       # IP-Adapter FaceID nodes + models (~2 GB)
#   ./install_ai_models.sh all
set -euo pipefail
cd "$(dirname "$0")"

COMFY="${COMFYUI_ROOT:-../ComfyUI}"
CN_DIR="$COMFY/models/controlnet"
CUSTOM="$COMFY/custom_nodes"

dl() {  # dl <url> <dest>
  local url="$1" dest="$2"
  if [ -f "$dest" ]; then echo "  ✓ exists: $(basename "$dest")"; return; fi
  echo "  ↓ $(basename "$dest")"
  curl -L -C - --fail -o "$dest.part" "$url" && mv "$dest.part" "$dest"
}

install_controlnet() {
  mkdir -p "$CN_DIR"
  echo "ControlNet SDXL models -> $CN_DIR"
  # Union (structure: canny / depth / soft-edge / pose) — ~2.5 GB
  dl "https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors" \
     "$CN_DIR/controlnet-union-sdxl-promax.safetensors"
  # Tile (texture / high-frequency detail / recolor) — ~2.5 GB
  dl "https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors" \
     "$CN_DIR/controlnet-tile-sdxl.safetensors"
  echo "ControlNet done. (engine maps: union→canny/depth/softedge, tile→tile)"
}

install_ipadapter() {
  echo "IP-Adapter FaceID (identity via diffusion) -> $CUSTOM + models"
  if [ ! -d "$CUSTOM/ComfyUI_IPAdapter_plus" ]; then
    git clone --depth 1 https://github.com/cubiq/ComfyUI_IPAdapter_plus "$CUSTOM/ComfyUI_IPAdapter_plus"
  else echo "  ✓ ComfyUI_IPAdapter_plus present"; fi
  mkdir -p "$COMFY/models/ipadapter" "$COMFY/models/clip_vision"
  dl "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin" \
     "$COMFY/models/ipadapter/ip-adapter-faceid_sdxl.bin"
  dl "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" \
     "$COMFY/models/clip_vision/CLIP-ViT-H-14.safetensors"
  echo "IP-Adapter done. (insightface for FaceID is already installed via pip)"
  echo "NOTE: restart ComfyUI so the new custom node loads."
}

case "${1:-controlnet}" in
  controlnet) install_controlnet ;;
  ipadapter)  install_ipadapter ;;
  all)        install_controlnet; install_ipadapter ;;
  *) echo "usage: $0 [controlnet|ipadapter|all]"; exit 2 ;;
esac

echo
echo "Restart ComfyUI, then verify which upgrades are now active:"
echo "  curl -s localhost:8189/comfyui/capabilities | python3 -m json.tool"
