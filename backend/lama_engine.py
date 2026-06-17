"""
Inpaint engine for the editable-AI-image editor.

Primary: LaMa (big-lama.pt) loaded directly via torch.jit. We bypass
lama-cleaner's ModelManager because this venv's huggingface_hub no longer
exposes `cached_download`, which lama-cleaner 1.2.5 imports at load time. The
LaMa weights are already cached by a previous lama-cleaner run, so we load the
TorchScript module straight from disk.

Fallback: OpenCV cv2.inpaint (Telea), so the /inpaint endpoint always works
even if the LaMa model cannot be loaded.
"""

import os

import cv2
import numpy as np
import torch

LAMA_JIT_PATH = os.environ.get(
    "LAMA_JIT_PATH",
    os.path.expanduser("~/.cache/torch/hub/checkpoints/big-lama.pt"),
)
PAD_MOD = 8


def _ceil_to(value, mod):
    return ((value + mod - 1) // mod) * mod


class InpaintEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.engine = "opencv"  # overwritten to "lama" on successful load
        self._load_lama()

    def _load_lama(self):
        if not os.path.exists(LAMA_JIT_PATH):
            print(f"[inpaint] LaMa weights not found at {LAMA_JIT_PATH}; using OpenCV.")
            return
        try:
            model = torch.jit.load(LAMA_JIT_PATH, map_location=self.device)
            model.eval()
            self.model = model
            self.engine = "lama"
            print(f"[inpaint] LaMa loaded on {self.device}.")
        except Exception as exc:  # noqa: BLE001 - fall back on any load failure
            print(f"[inpaint] LaMa load failed ({exc}); using OpenCV.")
            self.model = None
            self.engine = "opencv"

    @torch.no_grad()
    def inpaint(self, image_rgb, mask):
        """
        image_rgb : HxWx3 uint8 (RGB)
        mask      : HxW uint8 (255 = region to fill)
        returns   : HxWx3 uint8 (RGB), inpainted
        """
        if self.model is None:
            bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            binmask = (mask > 0).astype("uint8") * 255
            res = cv2.inpaint(bgr, binmask, 7, cv2.INPAINT_TELEA)
            return cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

        h, w = image_rgb.shape[:2]
        ph, pw = _ceil_to(h, PAD_MOD), _ceil_to(w, PAD_MOD)

        img = np.pad(image_rgb, ((0, ph - h), (0, pw - w), (0, 0)), mode="reflect")
        msk = np.pad(mask, ((0, ph - h), (0, pw - w)), mode="constant")

        img_t = (
            torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(self.device)
        )
        msk_t = (
            torch.from_numpy((msk > 0).astype("float32")).unsqueeze(0).unsqueeze(0).to(self.device)
        )

        out = self.model(img_t, msk_t)
        out = out[0].permute(1, 2, 0).detach().cpu().numpy()
        if out.max() <= 1.5:  # model emitted 0..1 rather than 0..255
            out = out * 255.0
        out = np.clip(out, 0, 255).astype("uint8")
        return out[:h, :w, :]
