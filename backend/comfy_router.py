"""
FastAPI router exposing the ComfyUI bridge.

  GET  /comfyui/status               ComfyUI reachability + basic info.
  POST /comfyui/{replace,background,remove,recolor,clothes,hair,person}
                                     object-level AI edits (shared engine).
  POST /comfyui/identity             impose a reference face's identity.
  POST /comfyui/lighting             CPU lighting/shadow harmonization.
  GET  /comfyui/identity/capabilities, /comfyui/lighting/config

Phase-2 refactor: the identical 6-clause exception→HTTP mapping that every edit
endpoint repeated now lives once in `_comfy_errors`; the repeated sampler fields
live once in `_SamplerOptions`. Status codes and response shapes are unchanged.

The router carries its own `/comfyui` prefix and is mounted by the standalone
bridge app (comfy_app.py), so the existing inpaint service (app.py) is untouched.
"""

from contextlib import contextmanager

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from comfy_client import ComfyUIClient, ComfyUIUnavailable, ComfyUIRequestError, ComfyUIError
from comfy_replace import replace_object, ReplaceError
from comfy_background import replace_background, BackgroundError
from comfy_remove import remove_object, RemoveError
from comfy_recolor import recolor_object, RecolorError
from comfy_clothes import change_clothes, ClothesError
from comfy_hair import change_hair, HairError
from comfy_person import replace_person, PersonError
from comfy_face import face_edit, FaceError
from comfy_logo import logo_edit, LogoError
from comfy_identity import identity_preserve, IdentityError
from comfy_lighting import harmonize as lighting_harmonize

router = APIRouter(prefix="/comfyui", tags=["comfyui"])

# One shared client; constructing it opens no sockets (httpx.Client is lazy).
_client = ComfyUIClient()


@contextmanager
def _comfy_errors(domain_error=None, domain_status=404):
    """Map the bridge's exception taxonomy to HTTP responses (one place).

    domain_error → domain_status (a bad request: object not found, etc.).
    ComfyUI transport/validation failures → 503/502/504. Anything else propagates
    (FastAPI returns 500), so genuine bugs are never masked as 4xx.
    """
    try:
        yield
    except ComfyUIUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"ComfyUI is offline: {exc}")
    except ComfyUIRequestError as exc:
        # ComfyUI was reached but rejected the graph — most often the SDXL
        # checkpoint isn't installed. Surface its payload so the cause is clear.
        raise HTTPException(status_code=502, detail={"message": str(exc), "comfyui": exc.payload})
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc))
    except ComfyUIError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        if domain_error is not None and isinstance(exc, domain_error):
            raise HTTPException(status_code=domain_status, detail=str(exc))
        raise


class _SamplerOptions(BaseModel):
    """Sampler/prompt overrides shared by every SDXL edit endpoint."""
    prompt: str | None = None
    negative: str | None = None
    denoise: float | None = None
    steps: int | None = None
    cfg: float | None = None
    seed: int | None = None
    sampler: str | None = None
    scheduler: str | None = None
    style: str | None = None       # "preserve" (default) / "photoreal" / "none"
    intensity: float | None = None  # 0..1 edit strength (maps to denoise)
    # --- Phase 14 quality controls (all optional; default off = unchanged) ---
    harmonize: bool | None = None        # auto lighting/colour/edge harmonization
    harmonizeStrength: float | None = None
    n: int | None = None                 # best-of-N candidates (capped at N_MAX)
    evaluator: bool | None = None        # attach the multi-dimensional quality score
    auto: bool | None = None             # convenience: harmonize + best-of-2 + score


def _resolve_quality(options):
    """Expand the `auto` convenience flag into the individual quality controls.
    Explicit per-flag values always win over `auto`. Mutates and returns options."""
    if options.pop("auto", None):
        options.setdefault("harmonize", True)
        options.setdefault("n", 2)
        options.setdefault("evaluator", True)
    return options


@router.get("/status")
def status():
    """ComfyUI connectivity + basic info. Always 200; `online` flags reachability."""
    return _client.status()


@router.get("/capabilities")
def capabilities():
    """Which Phase-3 quality upgrades are active (ControlNet / inpaint / soft / FaceID).

    Drives the auto-detect+fallback design: edits use each upgrade only when its
    model/node is installed; this endpoint reports what's currently available.
    """
    import comfy_capabilities
    return comfy_capabilities.summary(_client)


class ReplaceRequest(_SamplerOptions):
    objectId: int
    replacement: str           # data URL (or bare base64) of the uploaded image


@router.post("/replace")
def replace(req: ReplaceRequest):
    """Run the SDXL object-replacement workflow and return the new layer patch.

    -> { objectId, x, y, w, h, png } where `png` is a base64 PNG (RGBA) to drop
    onto the object's layer at (x, y). The rest of the canvas is never touched.
    """
    options = req.model_dump(exclude_none=True)
    options.pop("objectId", None)
    options.pop("replacement", None)
    _resolve_quality(options)
    with _comfy_errors(ReplaceError, 404):
        return replace_object(req.objectId, req.replacement, options=options, client=_client)


class BackgroundRequest(_SamplerOptions):
    replacement: str | None = None   # uploaded backdrop — required only for mode="replace"
    mode: str | None = None          # replace (default) / generate / blur / color / remove
    targetColor: str | list[int] | None = None   # for mode="color"
    blur: float | None = None        # 0..1 strength for mode="blur"


@router.post("/background")
def background(req: BackgroundRequest):
    """Replace the background with the uploaded image, preserving the foreground.

    -> { x:0, y:0, w, h, png, full:true } where `png` is the full canvas (RGB)
    with the new backdrop; foreground objects are kept exactly.
    """
    options = req.model_dump(exclude_none=True)
    options.pop("replacement", None)
    with _comfy_errors(BackgroundError, 404):
        return replace_background(req.replacement, options=options, client=_client)


class RemoveRequest(_SamplerOptions):
    objectId: int


@router.post("/remove")
def remove(req: RemoveRequest):
    """Erase the selected object and fill its footprint with realistic background.

    -> { objectId, x, y, w, h, png } RGBA patch for the object's footprint only.
    """
    options = req.model_dump(exclude_none=True)
    options.pop("objectId", None)
    _resolve_quality(options)
    with _comfy_errors(RemoveError, 404):
        return remove_object(req.objectId, options=options, client=_client)


class RecolorRequest(_SamplerOptions):
    objectId: int
    targetColor: str | list[int]   # "#ff0000" / "rgb(255,0,0)" / [255,0,0]


@router.post("/recolor")
def recolor(req: RecolorRequest):
    """Recolour the selected object to `targetColor`, preserving texture/lighting.

    -> { objectId, x, y, w, h, png } RGBA patch for the object's footprint only.
    """
    options = req.model_dump(exclude_none=True)
    options.pop("objectId", None)
    options.pop("targetColor", None)
    _resolve_quality(options)
    with _comfy_errors(RecolorError, 404):
        return recolor_object(req.objectId, req.targetColor, options=options, client=_client)


class ClothesRequest(_SamplerOptions):
    objectId: int
    prompt: str                # required for clothes (overrides the optional base)


@router.post("/clothes")
def clothes(req: ClothesRequest):
    """Regenerate only the person's clothing from a prompt (face/hair/pose kept).

    -> { objectId, x, y, w, h, png } RGBA patch for the clothing region only.
    """
    options = req.model_dump(exclude_none=True)
    options.pop("objectId", None)
    options.pop("prompt", None)
    _resolve_quality(options)
    with _comfy_errors(ClothesError, 404):
        return change_clothes(req.objectId, req.prompt, options=options, client=_client)


class HairRequest(_SamplerOptions):
    objectId: int
    prompt: str                # required for hair (overrides the optional base)


@router.post("/hair")
def hair(req: HairRequest):
    """Regenerate only the person's hair from a prompt (face/clothing/bg kept).

    -> { objectId, x, y, w, h, png } RGBA patch for the hair region only.
    """
    options = req.model_dump(exclude_none=True)
    options.pop("objectId", None)
    options.pop("prompt", None)
    _resolve_quality(options)
    with _comfy_errors(HairError, 404):
        return change_hair(req.objectId, req.prompt, options=options, client=_client)


class PersonRequest(_SamplerOptions):
    objectId: int
    image: str | None = None    # data URL of an uploaded replacement person
    # prompt (inherited) OR image — at least one is required


@router.post("/person")
def person(req: PersonRequest):
    """Replace the selected person (uploaded image OR generated from a prompt).

    -> { objectId, mode, x, y, w, h, png } RGBA patch for the person footprint only.
    """
    options = req.model_dump(exclude_none=True)
    for k in ("objectId", "image", "prompt"):
        options.pop(k, None)
    _resolve_quality(options)
    with _comfy_errors(PersonError, 400):
        return replace_person(req.objectId, image=req.image, prompt=req.prompt, options=options, client=_client)


class FaceRequest(_SamplerOptions):
    objectId: int
    feature: str                 # beard / smile / age / glasses / skin
    direction: str | None = None  # age: "older" (default) / "younger"


@router.post("/face")
def face(req: FaceRequest):
    """Apply a facial edit (beard/smile/age/glasses/skin) to the selected face.

    -> { objectId, feature, x, y, w, h, png } RGBA patch for the edited face
    sub-region only. Everything outside it is preserved exactly.
    """
    options = req.model_dump(exclude_none=True)
    for k in ("objectId", "feature"):
        options.pop(k, None)
    _resolve_quality(options)
    with _comfy_errors(FaceError, 404):
        return face_edit(req.objectId, req.feature, options=options, client=_client)


class LogoRequest(_SamplerOptions):
    objectId: int
    feature: str                 # metallic / glass / emboss / transparent


@router.post("/logo")
def logo(req: LogoRequest):
    """Apply a logo finish (metallic/glass via SDXL, emboss/transparent via CPU).

    -> { objectId, feature, x, y, w, h, png } RGBA footprint patch only.
    """
    options = req.model_dump(exclude_none=True)
    for k in ("objectId", "feature"):
        options.pop(k, None)
    _resolve_quality(options)
    with _comfy_errors(LogoError, 404):
        return logo_edit(req.objectId, req.feature, options=options, client=_client)


class IdentityRequest(BaseModel):
    referenceImage: str          # data URL of the reference face
    targetImage: str             # data URL of the image to receive the identity
    mask: str | None = None      # optional data URL footprint mask
    strength: float | None = None


@router.get("/identity/capabilities")
def identity_capabilities():
    """Which identity method is active (insightface / ip-adapter / instantid)."""
    from identity_manager import capabilities
    return capabilities(_client)


@router.post("/identity")
def identity(req: IdentityRequest):
    """Impose the reference face's identity onto the target image.

    -> { png, method, faceFound, similarity, strength, capabilities }. If the
    target/mask carries an alpha footprint, `png` is an RGBA patch with it.
    """
    with _comfy_errors(IdentityError, 400):
        return identity_preserve(
            req.referenceImage, req.targetImage, mask=req.mask,
            strength=req.strength if req.strength is not None else 0.85,
            client=_client,
        )


class LightingRequest(BaseModel):
    patch: str                   # data URL of the AI patch (RGBA) or a full image
    reference: str               # data URL of the original scene (margined crop)
    patchLeft: int | None = 0    # patch offset within the reference
    patchTop: int | None = 0
    strength: float | None = None


@router.get("/lighting/config")
def lighting_config():
    """The active lighting harmonization config (read-only)."""
    from lighting_manager import config
    return config()


@router.post("/lighting")
def lighting(req: LightingRequest):
    """Harmonize a patch's lighting to the surrounding scene (CPU, no SDXL).

    -> { png, ok, report, capabilities }. Always 200: on failure `ok` is false
    and `png` is the original patch (never crashes / blocks editing).
    """
    options = {}
    if req.strength is not None:
        options["strength"] = req.strength
    return lighting_harmonize(req.patch, req.reference, req.patchLeft or 0, req.patchTop or 0, options)
