"""Depth estimation: AI model when installed, a fast heuristic otherwise.

AI engine: Depth Anything V2 (small) through Hugging Face `transformers`.
It runs on a normal laptop CPU in a few seconds; the model (~100 MB) is
downloaded on first use and cached.

Fallback engine: no downloads, no GPU. It "inflates" the subject outline
into a dome and adds shading detail from the photo. Good enough to test the
full workflow, but faces look much better with the AI engine.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image
from scipy import ndimage

AI_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def ai_depth_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _pipeline():
    from transformers import pipeline

    return pipeline("depth-estimation", model=AI_MODEL)


def estimate_depth(
    img: Image.Image, mask: np.ndarray | None = None, engine: str = "auto", faces=()
) -> tuple[np.ndarray, str]:
    """Return (depth 0..1 where 1 = closest to camera, engine name)."""
    if engine in ("auto", "ai") and ai_depth_available():
        try:
            out = _pipeline()(img)
            d = np.asarray(out["predicted_depth"].squeeze().cpu().numpy(), dtype=np.float64)
            d = np.asarray(Image.fromarray(d).resize(img.size, Image.BICUBIC), dtype=np.float64)
            return _normalise(d, mask), "Depth Anything V2 (AI)"
        except Exception as exc:  # network / model problems -> keep working
            if engine == "ai":
                raise RuntimeError(f"AI depth failed: {exc}") from exc
    return heuristic_depth(img, mask, faces), "fast heuristic (no AI)"


def heuristic_depth(img: Image.Image, mask: np.ndarray | None = None, faces=()) -> np.ndarray:
    gray = np.asarray(img.convert("L"), dtype=np.float64) / 255.0
    if mask is None:
        mask = np.ones_like(gray)
    hard = mask > 0.5
    dist = ndimage.distance_transform_edt(hard)
    if dist.max() > 0:
        dist /= dist.max()
    dome = np.sqrt(np.clip(dist * (2 - dist), 0, 1))
    # Lighter areas of a face (forehead, nose, cheeks) tend to face the camera.
    shade = ndimage.gaussian_filter(gray, max(1.0, min(gray.shape) / 150))
    shade = (shade - shade[hard].mean()) if hard.any() else shade * 0
    depth = dome * 0.8 + shade * 0.35
    # Faces are rounded and come towards the camera: add a soft ellipsoid
    # per detected face so cheeks, nose and forehead get real volume.
    if faces:
        h, w = gray.shape
        yy, xx = np.mgrid[0:h, 0:w]
        for x, y, fw, fh in faces:
            cx, cy = x + fw / 2, y + fh * 0.55
            d = ((xx - cx) / (fw * 0.55)) ** 2 + ((yy - cy) / (fh * 0.72)) ** 2
            depth += np.sqrt(np.clip(1 - d, 0, 1)) * 0.55
    return _normalise(depth, mask)


def _normalise(d: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    region = d[mask > 0.5] if mask is not None and (mask > 0.5).any() else d
    lo, hi = np.percentile(region, 1), np.percentile(region, 99.5)
    out = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    if mask is not None:
        out *= mask
    return out
