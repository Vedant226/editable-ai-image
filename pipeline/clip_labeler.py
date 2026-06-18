"""
CLIP zero-shot labeling over the taxonomy (one evidence source for fusion).

Uses transformers CLIP (openai/clip-vit-base-patch32; weights download once).
Returns a probability distribution over the taxonomy using CLIP's own
logit_scale so the top probability is meaningful (not diluted across 50 labels).
"""

import os

import torch
from transformers import CLIPModel, CLIPProcessor

from . import config as C

MODEL_NAME = "openai/clip-vit-base-patch32"
# CLIP runs on CPU by default: it is small/fast there and leaves the limited GPU
# entirely to SAM's encoder (which OOMs an ~8GB card if CLIP is also resident).
DEVICE = os.environ.get("CLIP_DEVICE", "cpu")
_state = {}


def load_clip():
    if "model" not in _state:
        model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
        proc = CLIPProcessor.from_pretrained(MODEL_NAME)
        prompts = [C.CLIP_PROMPTS.get(c, f"a photo of a {c}") for c in C.TAXONOMY]
        inputs = proc(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            tf = model.get_text_features(**inputs)
        _state["model"] = model
        _state["proc"] = proc
        _state["text_feats"] = torch.nn.functional.normalize(tf, dim=-1)
        _state["logit_scale"] = model.logit_scale.exp().detach()
    return _state


@torch.no_grad()
def classify(pil_image):
    s = load_clip()
    inputs = s["proc"](images=pil_image, return_tensors="pt").to(DEVICE)
    feat = torch.nn.functional.normalize(s["model"].get_image_features(**inputs), dim=-1)
    logits = (s["logit_scale"] * (feat @ s["text_feats"].T)).squeeze(0)
    probs = torch.softmax(logits, dim=-1)
    return {cat: float(probs[i]) for i, cat in enumerate(C.TAXONOMY)}
