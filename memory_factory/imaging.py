"""Image helpers: loading, background removal, masks and text rendering."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage

FONT_CANDIDATES = [
    # Windows
    "C:/Windows/Fonts/georgiab.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    # macOS
    "/Library/Fonts/Georgia Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def load_image(src) -> Image.Image:
    """Open a path / PIL image / numpy array, fix phone rotation, return RGB."""
    if isinstance(src, Image.Image):
        img = src
    elif isinstance(src, np.ndarray):
        img = Image.fromarray(src)
    else:
        img = Image.open(src)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def limit_size(img: Image.Image, max_side: int = 1600) -> Image.Image:
    if max(img.size) <= max_side:
        return img
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def to_gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float64) / 255.0


# --------------------------------------------------------------------------
# Background removal
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _rembg_session():
    from rembg import new_session  # optional dependency

    return new_session("u2net_human_seg")


def rembg_available() -> bool:
    try:
        import rembg  # noqa: F401

        return True
    except Exception:
        return False


def subject_mask(img: Image.Image, use_ai: bool = True) -> tuple[np.ndarray, str]:
    """Return (mask in 0..1, engine name) for the person/people in a photo."""
    if use_ai and rembg_available():
        try:
            from rembg import remove

            cut = remove(img, session=_rembg_session())
            alpha = np.asarray(cut.split()[-1], dtype=np.float64) / 255.0
            if alpha.mean() > 0.02:
                return alpha, "rembg (AI)"
        except Exception:
            pass
    return _fallback_subject_mask(img), "simple (no AI)"


def _fallback_subject_mask(img: Image.Image) -> np.ndarray:
    """No-AI subject mask.

    Works well for photos taken against a plain backdrop (recommended in-shop
    setup): anything that differs from the border colour is the subject. If
    that fails (busy background) a soft centred oval is used instead.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    border = np.concatenate([arr[:4].reshape(-1, 3), arr[-4:].reshape(-1, 3),
                             arr[:, :4].reshape(-1, 3), arr[:, -4:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    spread = np.median(np.abs(border - bg).sum(axis=1))
    if spread < 0.12:  # border is fairly uniform -> plain backdrop
        diff = np.abs(arr - bg).sum(axis=2)
        mask = diff > max(0.15, spread * 4)
        k = max(3, int(min(mask.shape) * 0.01))
        mask = ndimage.binary_opening(mask, structure=np.ones((k, k)))
        mask = ndimage.binary_closing(mask, structure=np.ones((k, k)), iterations=2)
        labels, count = ndimage.label(mask)
        if count:
            sizes = ndimage.sum(mask, labels, range(1, count + 1))
            mask = np.isin(labels, np.flatnonzero(sizes >= 0.05 * sizes.max()) + 1)
            mask = ndimage.binary_fill_holes(mask)
            if 0.05 < mask.mean() < 0.9:
                return ndimage.gaussian_filter(mask.astype(float), 1.0)
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w]
    d = ((xx - w / 2) / (w * 0.48)) ** 2 + ((yy - h * 0.55) / (h * 0.55)) ** 2
    return np.clip((1.15 - d) / 0.3, 0, 1)


# --------------------------------------------------------------------------
# Faces
# --------------------------------------------------------------------------

def detect_faces(img: Image.Image, mask: np.ndarray | None = None) -> list[tuple[int, int, int, int]]:
    """Frontal faces as (x, y, w, h), largest first. Empty list if OpenCV is missing.

    Uses OpenCV's bundled Haar cascade (no download). Detections outside the
    subject mask (faces "seen" in trees, posters...) and tiny ones are dropped.
    """
    try:
        import cv2
    except ImportError:
        return []
    gray = np.asarray(img.convert("L"))
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        return []
    min_side = max(24, int(min(gray.shape) * 0.06))
    found = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=6,
                                     minSize=(min_side, min_side))
    faces = sorted([tuple(int(v) for v in f) for f in found], key=lambda f: -f[2] * f[3])
    if not faces:
        return []
    biggest = faces[0][2]
    keep = []
    for x, y, w, h in faces:
        if w < 0.45 * biggest:
            continue
        if mask is not None and mask[min(y + h // 2, mask.shape[0] - 1),
                                     min(x + w // 2, mask.shape[1] - 1)] < 0.5:
            continue
        keep.append((x, y, w, h))
    return keep[:6]


def portrait_crop_box(size: tuple[int, int], mask: np.ndarray, faces, aspect: float = 1.1):
    """Head-and-shoulders crop (x0, y0, x1, y1) around the detected faces.

    From a little above the heads (hair, caps) to under the chins
    (shoulders), widened to `aspect` (width / height).
    """
    w_img, h_img = size
    if faces:
        xs0 = min(f[0] for f in faces)
        ys0 = min(f[1] for f in faces)
        xs1 = max(f[0] + f[2] for f in faces)
        ys1 = max(f[1] + f[3] for f in faces)
        fh = float(np.median([f[3] for f in faces]))
        x0, x1 = xs0 - 0.45 * fh, xs1 + 0.45 * fh
        y0, y1 = ys0 - 0.5 * fh, ys1 + 0.8 * fh
    else:  # no face found: top part of the subject
        ys, xs = np.nonzero(mask > 0.5)
        if len(xs) == 0:
            return 0, 0, w_img, h_img
        x0, x1, y0 = xs.min(), xs.max(), ys.min()
        y1 = y0 + (x1 - x0) / aspect
    # Widen or heighten to the target aspect ratio, centred.
    bw, bh = x1 - x0, y1 - y0
    if bw / bh < aspect:
        grow = (bh * aspect - bw) / 2
        x0, x1 = x0 - grow, x1 + grow
    else:
        y1 = y0 + bw / aspect
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w_img, int(x1)), min(h_img, int(y1))
    return x0, y0, x1, y1


def face_weight_map(shape, faces, spread: float = 1.35) -> np.ndarray:
    """0..1 map that is 1 over faces + hair and fades out around them."""
    h, w = shape
    out = np.zeros((h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    for x, y, fw, fh in faces:
        cx, cy = x + fw / 2, y + fh * 0.45
        d = ((xx - cx) / (fw * 0.5 * spread)) ** 2 + ((yy - cy) / (fh * 0.62 * spread)) ** 2
        out = np.maximum(out, np.clip(1.6 - d, 0, 1))
    return ndimage.gaussian_filter(out, max(1.0, min(h, w) / 80))


def crop_to_mask(img: Image.Image, mask: np.ndarray, pad: float = 0.06):
    """Crop image+mask to the subject's bounding box with some padding."""
    ys, xs = np.nonzero(mask > 0.5)
    if len(xs) == 0:
        return img, mask
    h, w = mask.shape
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
    x0, y0 = max(0, x0 - px), max(0, y0 - py)
    x1, y1 = min(w, x1 + px), min(h, y1 + py)
    return img.crop((x0, y0, x1, y1)), mask[y0:y1, x0:x1]


# --------------------------------------------------------------------------
# Drawing (paper) segmentation
# --------------------------------------------------------------------------

def drawing_mask(img: Image.Image, min_area_frac: float = 0.002) -> np.ndarray:
    """Separate a child's drawing from the white/cream paper.

    Ink and crayon are darker and/or more colourful than paper. We threshold
    on a mix of darkness and saturation, close small gaps, fill enclosed
    areas (the inside of an outlined shape counts as the shape) and drop
    specks of dust.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    gray = arr.mean(axis=2)
    sat = arr.max(axis=2) - arr.min(axis=2)
    paper = np.percentile(gray, 90)
    score = np.clip((paper - gray) * 2.0, 0, 1) + sat * 1.5
    thr = max(0.12, _otsu(score))
    mask = score > thr

    k = max(3, int(min(mask.shape) * 0.01))
    mask = ndimage.binary_closing(mask, structure=np.ones((k, k)), iterations=2)
    mask = ndimage.binary_fill_holes(mask)

    labels, count = ndimage.label(mask)
    if count:
        sizes = ndimage.sum(mask, labels, range(1, count + 1))
        keep = np.flatnonzero(sizes >= min_area_frac * mask.size) + 1
        mask = np.isin(labels, keep)
    return mask


def _otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(values.ravel(), bins=256)
    p = hist.astype(float) / max(hist.sum(), 1)
    omega = np.cumsum(p)
    centers = (edges[:-1] + edges[1:]) / 2
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    sigma_b = np.nan_to_num(sigma_b)
    return float(centers[int(np.argmax(sigma_b))])


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------

def find_font(custom: str | None = None) -> str | None:
    for candidate in [custom, os.environ.get("MF_FONT"), *FONT_CANDIDATES]:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def get_font(size_px: int, custom: str | None = None) -> ImageFont.ImageFont:
    path = find_font(custom)
    if path:
        return ImageFont.truetype(path, size_px)
    return ImageFont.load_default(size=size_px)


def text_strip(text: str, height_px: int, font_path: str | None = None) -> np.ndarray:
    """One line of text, `height_px` tall, cropped tight left/right (0..1)."""
    text = (text or "").strip()
    if not text or height_px < 4:
        return np.zeros((max(height_px, 1), 1))
    font = get_font(height_px, font_path)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    # Size the font so capital letters + descenders fill the strip height.
    l, t, r, b = probe.textbbox((0, 0), "Hg" + text, font=font)
    font = get_font(max(6, int(height_px * height_px / max(b - t, 1))), font_path)
    l, t, r, b = probe.textbbox((0, 0), text, font=font)
    canvas = Image.new("L", (r - l + 4, height_px), 0)
    lt, tt, rt, bt = probe.textbbox((0, 0), "Hg", font=font)
    ImageDraw.Draw(canvas).text((2 - l, (height_px - (bt - tt)) / 2 - tt), text, fill=255, font=font)
    return np.asarray(canvas, dtype=np.float64) / 255.0


def text_mask(
    text: str,
    width_px: int,
    height_px: int,
    font_path: str | None = None,
    max_font_px: int | None = None,
) -> np.ndarray:
    """Render one line of text centred, as large as fits the box (0..1 mask)."""
    canvas = Image.new("L", (width_px, height_px), 0)
    text = (text or "").strip()
    if not text:
        return np.zeros((height_px, width_px))
    size = max_font_px or height_px
    draw = ImageDraw.Draw(canvas)
    while size > 6:
        font = get_font(size, font_path)
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        if r - l <= width_px * 0.96 and b - t <= height_px * 0.92:
            break
        size -= 2
    draw.text(
        ((width_px - (r - l)) / 2 - l, (height_px - (b - t)) / 2 - t),
        text,
        fill=255,
        font=font,
    )
    return np.asarray(canvas, dtype=np.float64) / 255.0
