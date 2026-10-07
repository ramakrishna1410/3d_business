"""2D previews: realistic shaded renders, proof sheets and painting guides.

These images are what the customer approves (e.g. sent on WhatsApp), so they
show the piece lit from the side like a real object, not a grey depth map.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .imaging import get_font

MATERIALS = {
    "antique brass": ((181, 140, 62), (255, 226, 150), 0.55),
    "gold": ((196, 150, 60), (255, 232, 160), 0.7),
    "bronze": ((140, 86, 45), (240, 180, 120), 0.45),
    "silver": ((150, 150, 155), (250, 250, 255), 0.6),
    "white resin": ((215, 212, 205), (255, 255, 255), 0.15),
    "grey resin": ((120, 122, 128), (220, 222, 228), 0.2),
}


def shade(height_mm: np.ndarray, pixel_mm: float, light_deg=(315.0, 40.0)) -> np.ndarray:
    """Lambert shading with light from top-left (0..1)."""
    gy, gx = np.gradient(height_mm, pixel_mm)
    nx, ny, nz = -gx, gy, np.ones_like(gx)
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    az, el = np.radians(light_deg[0]), np.radians(light_deg[1])
    lx, ly, lz = np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)
    lam = (nx * lx + ny * ly + nz * lz) / norm
    return np.clip(lam, 0, 1)


def render_material(
    height_mm: np.ndarray,
    pixel_mm: float,
    mask: np.ndarray | None = None,
    material: str = "antique brass",
    background=(245, 242, 236),
) -> Image.Image:
    dark, light, spec_strength = MATERIALS.get(material, MATERIALS["antique brass"])
    lam = shade(height_mm, pixel_mm)
    # Ambient occlusion-ish darkening in recesses (antique finish look).
    from scipy import ndimage

    cavity = height_mm - ndimage.gaussian_filter(height_mm, 3)
    cavity = np.clip(0.5 + cavity / (np.abs(cavity).max() + 1e-9), 0, 1)
    t = np.clip(0.15 + 0.85 * lam * (0.6 + 0.4 * cavity), 0, 1)
    spec = np.clip(lam, 0, 1) ** 24 * spec_strength
    dark, light = np.array(dark, float), np.array(light, float)
    rgb = dark[None, None] + (light - dark)[None, None] * t[..., None]
    rgb = np.clip(rgb + spec[..., None] * 255, 0, 255)
    if mask is not None:
        m = np.asarray(mask, float)[..., None]
        rgb = rgb * m + np.array(background, float)[None, None] * (1 - m)
    return Image.fromarray(rgb.astype(np.uint8))


def render_coloured(height_mm, pixel_mm, colour_img: Image.Image, mask=None,
                    background=(245, 242, 236)) -> Image.Image:
    """Shade the height map using the original drawing's colours (painted look)."""
    lam = shade(height_mm, pixel_mm)
    col = np.asarray(colour_img.resize(height_mm.shape[::-1]).convert("RGB"), float)
    rgb = col * (0.35 + 0.75 * lam[..., None])
    rgb = np.clip(rgb, 0, 255)
    if mask is not None:
        m = np.asarray(mask, float)[..., None]
        rgb = rgb * m + np.array(background, float)[None, None] * (1 - m)
    return Image.fromarray(rgb.astype(np.uint8))


def proof_sheet(render: Image.Image, title: str, lines: list[str],
                brand: str = "Memory Factory") -> Image.Image:
    """Customer approval card: render + order details, sized for WhatsApp."""
    w = 1080
    r = render.copy()
    side = w - 240
    scale = min(side / r.width, side / r.height)
    r = r.resize((int(r.width * scale), int(r.height * scale)), Image.LANCZOS)
    small = get_font(28)
    h = 200 + r.height + 30 + 42 * len(lines) + 110
    sheet = Image.new("RGB", (w, h), (252, 249, 243))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle([0, 0, w, 110], fill=(92, 30, 36))
    draw.text((40, 30), brand, font=get_font(48), fill=(245, 214, 140))
    draw.text((40, 130), title, font=get_font(40), fill=(60, 40, 30))
    sheet.paste(r, ((w - r.width) // 2, 200))
    y = 230 + r.height
    for line in lines:
        draw.text((40, y), line, font=small, fill=(70, 60, 55))
        y += 42
    note = get_font(24)
    draw.text((40, h - 90), "Please reply APPROVE or send your changes.", font=note,
              fill=(92, 30, 36))
    draw.text((40, h - 55), "This is a computer preview - the handmade finish may vary slightly.",
              font=note, fill=(120, 110, 100))
    return sheet


def painting_guide(colour_img: Image.Image, mask: np.ndarray, n_colours: int = 8) -> Image.Image:
    """Palette of the drawing's main colours so the painter can match them."""
    arr = np.asarray(colour_img.convert("RGB"))
    m = np.asarray(mask) > 0.5
    pixels = arr[m]
    if len(pixels) == 0:
        pixels = arr.reshape(-1, 3)
    sample = pixels[:: max(1, len(pixels) // 20000)]
    strip = Image.fromarray(sample.reshape(-1, 1, 3).astype(np.uint8))
    q = strip.quantize(colors=n_colours, method=Image.Quantize.MEDIANCUT)
    pal = np.array(q.getpalette()[: n_colours * 3]).reshape(-1, 3)
    counts = np.bincount(np.asarray(q).ravel(), minlength=n_colours)
    # Merge near-identical shades so the painter sees distinct paints only.
    kept: list[int] = []
    merged = counts.astype(float)
    for idx in np.argsort(-counts):
        if counts[idx] == 0:
            continue
        near = [k for k in kept if np.abs(pal[k].astype(int) - pal[idx].astype(int)).sum() < 60]
        if near:
            merged[near[0]] += merged[idx]
            merged[idx] = 0
        else:
            kept.append(idx)
    counts = merged
    order = [k for k in np.argsort(-counts) if counts[k] > 0]
    n_colours = len(order)

    sw, pad = 150, 20
    img = Image.new("RGB", (pad + n_colours * (sw + pad), sw + 110), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = get_font(22)
    draw.text((pad, 8), "Painting guide - main colours (most used first)", font=font,
              fill=(40, 40, 40))
    for k, idx in enumerate(order):
        if counts[idx] == 0:
            continue
        c = tuple(int(v) for v in pal[idx])
        x = pad + k * (sw + pad)
        draw.rectangle([x, 45, x + sw, 45 + sw - 50], fill=c, outline=(0, 0, 0))
        pct = 100 * counts[idx] / counts.sum()
        draw.text((x, sw + 5), f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}", font=font, fill=(0, 0, 0))
        draw.text((x, sw + 35), f"{pct:.0f}%", font=font, fill=(90, 90, 90))
    return img
