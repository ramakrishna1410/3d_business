"""Check a customer photo BEFORE spending Tripo credits, and make the Tripo-ready close-up.

Tripo makes the shape of a face from the light and shadow in the picture. It
gets the likeness right only when the face is big, sharp, evenly lit and looking
at the camera - so the photo is checked first (free, offline), and then cropped
to a square head-and-shoulders close-up on a white background (1024 px), where
the face fills about half of the width.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

from . import imaging

GOOD, WARNING, BAD = "good", "warning", "bad"
MIN_FACE_PX = 150          # below this Tripo cannot see the face -> generic result
GOOD_FACE_PX = 220
CROP_PX = 1024


@dataclass
class PhotoCheck:
    level: str                                  # good / warning / bad
    messages: list[str] = field(default_factory=list)
    crop: Image.Image | None = None             # Tripo-ready close-up
    face_px: int = 0
    faces: int = 0

    @property
    def ok(self) -> bool:
        """Good enough to spend Tripo credits on."""
        return self.level != BAD

    def as_markdown(self) -> str:
        head = {GOOD: "✅ **Photo is good** - ready for Tripo.",
                WARNING: "⚠️ **Usable, but check the points below** (a better photo gives a better face).",
                BAD: "❌ **Don't use this photo** - it would waste Tripo credits."}[self.level]
        return "\n".join([head, ""] + [f"- {m}" for m in self.messages])


def _eyes(gray: np.ndarray, face) -> list[tuple[float, float]]:
    try:
        import cv2
    except ImportError:
        return []
    x, y, w, h = face
    cc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    roi = gray[y:y + int(h * 0.6), x:x + w].astype(np.uint8)
    found = cc.detectMultiScale(roi, 1.05, 4, minSize=(max(8, w // 10), max(8, w // 10)))
    big = sorted(found, key=lambda e: -e[2])[:2]
    return sorted((ex + ew / 2, ey + eh / 2) for ex, ey, ew, eh in big)


def _sharpness(gray: np.ndarray, face) -> float:
    """Edge energy of the face resized to 256 px (independent of photo size)."""
    import cv2

    x, y, w, h = face
    roi = cv2.resize(gray[y:y + h, x:x + w].astype(np.uint8), (256, 256), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(roi.astype(float), cv2.CV_64F).var())


def tripo_crop(img: Image.Image, face, white_background: bool = True) -> Image.Image:
    """Square head-and-shoulders close-up (face ~45% of the width), white background, 1024 px."""
    x, y, w, h = face
    side = int(2.1 * w)
    cx = x + w / 2
    left, top = int(cx - side / 2), int(y - 0.6 * h)     # head centred, shoulders at the bottom
    if white_background:
        mask, _ = imaging.subject_mask(img)
        rgb = np.asarray(img.convert("RGB"), float)
        m = np.clip(mask, 0, 1)[..., None]
        img = Image.fromarray((rgb * m + 255 * (1 - m)).astype(np.uint8))
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(img.crop((max(0, left), max(0, top), min(img.width, left + side),
                           min(img.height, top + side))), (max(0, -left), max(0, -top)))
    return canvas.resize((CROP_PX, CROP_PX), Image.LANCZOS)


def check_photo(photo, white_background: bool = True) -> PhotoCheck:
    img = ImageOps.exif_transpose(imaging.load_image(photo)).convert("RGB")
    faces = imaging.detect_faces(img, None)
    if not faces:
        return PhotoCheck(BAD, ["No face found. Use a clear photo with the face looking at the camera "
                                "(no sunglasses, no mask, face not too small)."])
    face = faces[0]
    x, y, w, h = face
    gray = np.asarray(img.convert("L"), float)
    roi = gray[y:y + h, x:x + w] / 255
    bad, warn = [], []

    if len(faces) > 1:
        warn.append(f"{len(faces)} faces in the photo - the biggest one was cropped. Check that only "
                    "this person is in the close-up (a single-person photo is better).")
    if w < MIN_FACE_PX:
        bad.append(f"Face is only {w} px wide - too small for a good likeness. Send the original photo "
                   "(not WhatsApp-forwarded) or a closer photo.")
    elif w < GOOD_FACE_PX:
        warn.append(f"Face is {w} px wide - OK, but {GOOD_FACE_PX}+ px gives more detail "
                    "(use the original photo, a closer shot).")
    mean, std = float(roi.mean()), float(roi.std())
    if mean < 0.22:
        bad.append("Face is too dark. Use a photo in daylight or good indoor light.")
    elif mean > 0.85:
        warn.append("Face looks overexposed (too bright / flash). Details may be lost.")
    if std < 0.08:
        warn.append("Very flat, low-contrast face - the 3D face may come out smooth and generic.")
    half = abs(roi[:, : w // 2].mean() - roi[:, w // 2:].mean())
    if half > 0.25:
        warn.append("Strong light from one side (half the face in shadow). Even, soft light is better.")
    eyes = _eyes(gray, face)
    if len(eyes) < 2:
        warn.append("Eyes not clearly visible (sunglasses, cap shadow, closed eyes, or turned head?).")
    else:
        yaw = (np.mean([e[0] for e in eyes]) - w / 2) / w
        tilt = abs(eyes[1][1] - eyes[0][1]) / max(1.0, abs(eyes[1][0] - eyes[0][0]))
        if abs(yaw) > 0.1:
            warn.append("Head is turned to the side - a straight front view gives the best likeness.")
        if tilt > 0.25:
            warn.append("Head is tilted - a straight, upright head works best.")
    try:
        if _sharpness(gray, face) < 25:
            warn.append("Photo looks blurry. Use a sharp, in-focus photo.")
    except Exception:
        pass

    level = BAD if bad else (WARNING if warn else GOOD)
    msgs = bad + warn or [f"One face, {w} px wide, sharp, evenly lit, looking at the camera."]
    crop = tripo_crop(img, face, white_background) if level != BAD or w >= MIN_FACE_PX else None
    return PhotoCheck(level, msgs, crop, int(w), len(faces))
