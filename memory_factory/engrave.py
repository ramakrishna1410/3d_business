"""Engraved text for the Royal Chess pieces: a name on the front of the stand and a short
message + date under the base. The text is cut into the solid (part of the STL), so the
print shop prints it as is.

Fonts are bundled in assets/fonts (SIL Open Font License - free for commercial use).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
# label -> (file, kind). kind sets the smallest readable letter height.
FONTS = {
    "Script (Great Vibes)": ("GreatVibes-Regular.ttf", "script"),
    "Royal capitals (Cinzel)": ("Cinzel.ttf", "caps"),
    "Elegant italic (Playfair)": ("PlayfairDisplay-Italic.ttf", "italic"),
}
DATE_FONT = ("Cinzel.ttf", "caps")           # dates always in clear capitals / numerals
MIN_NAME_MM = {"script": 4.0, "caps": 3.0, "italic": 3.5}
MIN_MSG_MM = {"script": 3.5, "caps": 2.5, "italic": 3.0}
MIN_DATE_MM = 2.2
NAME_DEPTH, UNDER_DEPTH = 0.7, 0.6           # mm cut into the surface
BOLD_MM = 0.1                                # thickens hairlines so they survive printing


@dataclass
class Placed:
    polys: list            # outlines in mm, centred on (0, 0)
    width: float
    height: float
    text: str


def outline(text: str, font: tuple[str, str], height_mm: float, px_per_mm: int = 40) -> Placed:
    """Text -> closed outlines (mm), ink height = height_mm, centred."""
    from PIL import Image, ImageDraw, ImageFont
    from scipy import ndimage
    from skimage import measure

    f = ImageFont.truetype(str(FONT_DIR / font[0]), 400)
    left, top, right, bottom = f.getbbox(text)
    im = Image.new("L", (right - left + 80, bottom - top + 80), 0)
    ImageDraw.Draw(im).text((40 - left, 40 - top), text, font=f, fill=255)
    a = np.asarray(im) > 127
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        raise ValueError(f'"{text}" has no printable letters in this font.')
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    k = height_mm * px_per_mm / a.shape[0]
    a = np.asarray(Image.fromarray(a.astype(np.uint8) * 255).resize(
        (max(1, int(a.shape[1] * k)), max(1, int(a.shape[0] * k))), Image.LANCZOS)) > 127
    a = ndimage.binary_dilation(a, iterations=max(1, int(round(BOLD_MM * px_per_mm))))
    a = np.pad(a, 2)
    w, h = a.shape[1] / px_per_mm, a.shape[0] / px_per_mm
    polys = [np.column_stack([c[:, 1] / px_per_mm - w / 2, h / 2 - c[:, 0] / px_per_mm])
             for c in measure.find_contours(a.astype(float), 0.5)]
    return Placed(polys, w, h, text)


def _fits(n_letters: int, width: float, limit: float) -> int:
    return max(1, int(n_letters * limit / max(width, 1e-6)))


def fit_name(name: str, font_label: str, h_start: float, max_width: float) -> Placed:
    """Largest letter height (<= h_start) whose width fits; error if it would be unreadable."""
    font = FONTS[font_label]
    p = outline(name, font, h_start)
    if p.width <= max_width:
        return p
    h = h_start * max_width / p.width
    h_min = MIN_NAME_MM[font[1]]
    if h < h_min:
        n = _fits(len(name), p.width * h_min / h_start, max_width)
        raise ValueError(f'Name "{name}" is too long for this piece and size: about {n} letters fit '
                         f"in this font (try a shorter name, capitals, or the couple-gift size).")
    return outline(name, font, h)


def fit_underside(lines: list[tuple[str, tuple[str, str], float, float]], radius: float) -> list[tuple[Placed, float]]:
    """lines: (text, font, start height, min height). Stacks them in a circle of `radius`,
    shrinking all together until every line fits its chord. Returns (outline, y centre)."""
    shrink = 1.0
    for _ in range(40):
        placed = [outline(t, f, h0 * shrink) for t, f, h0, _ in lines]
        gap = 0.35 * min(p.height for p in placed)
        total = sum(p.height for p in placed) + gap * (len(placed) - 1)
        y = total / 2
        out, ok = [], True
        for p in placed:
            yc = y - p.height / 2
            ext = max(abs(y), abs(y - p.height))
            chord = 2 * np.sqrt(max(radius ** 2 - ext ** 2, 0.0))
            ok &= p.width <= chord
            out.append((p, yc))
            y -= p.height + gap
        if ok:
            return out
        shrink *= 0.95
        for (t, f, h0, hmin), p in zip(lines, placed):
            if p.height * 0.95 < hmin:
                raise ValueError(f'Underside text "{t}" is too long for this base - shorten it '
                                 "(full-size letters up to ~14 characters per line on the couple-gift "
                                 "King, ~10 on the chess set; longer lines get smaller letters).")
    raise ValueError("Underside text does not fit - shorten it.")


def stem_tool(p: Placed, r_at, z_mid: float, depth: float = NAME_DEPTH):
    """Solid of the letters wrapped around the front (-y) of a round stem; subtract it."""
    import manifold3d as mf

    cs = mf.CrossSection([q.astype(np.float64) for q in p.polys], mf.FillRule.EvenOdd)
    sol = cs.extrude(depth + 1.5).translate([0, 0, -depth]).refine_to_length(0.25)

    def warp(v):
        z = z_mid + v[1]
        r0 = float(r_at(z))
        r, th = r0 + v[2], v[0] / r0
        return [r * np.sin(th), -r * np.cos(th), z]

    return sol.warp(warp)


def underside_tool(placed: list[tuple[Placed, float]], depth: float = UNDER_DEPTH):
    """Letters cut up into the flat bottom (z = 0). Laid out to read correctly when the piece
    is turned over (front edge of the piece at the top)."""
    import manifold3d as mf

    tool = None
    for p, yc in placed:
        cs = mf.CrossSection([np.column_stack([q[:, 0], -(q[:, 1] + yc)]) for q in p.polys],
                             mf.FillRule.EvenOdd)
        t = cs.extrude(depth + 1.0).translate([0, 0, -1.0])
        tool = t if tool is None else tool + t
    return tool
