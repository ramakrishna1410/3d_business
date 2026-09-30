"""Family chess set: classic pieces, each carrying a relief portrait medallion.

Every piece = a turned (lathe) body + its classic top (cross, crown, mitre,
battlements, horse head, ball) + an oval "cameo" on the front with the face
of the family member it represents. Pieces without a photo get an initial.

    King   = grandfather / father        Queen  = grandmother / mother
    Bishop = uncles, aunts               Knight = children
    Rook   = grandparents / elders       Pawn   = cousins, grandchildren, pets

All sizes in mm. Standard tournament proportions: king ~90 mm, board square 42 mm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from . import depth as depth_mod, imaging, mesh as mesh_mod, relief

PIECES = ["king", "queen", "bishop", "knight", "rook", "pawn"]


@dataclass
class PieceSpec:
    height: float
    base_r: float
    cameo_w: float      # medallion width
    cameo_h: float      # medallion height
    cameo_z: float      # centre height of the medallion
    body_r_at_cameo: float


SPECS = {
    "king":   PieceSpec(92, 17, 20, 25, 38, 8.5),
    "queen":  PieceSpec(84, 16, 19, 24, 36, 8.0),
    "bishop": PieceSpec(72, 15, 17, 21, 31, 7.5),
    "knight": PieceSpec(64, 15, 16, 20, 30, 6.0),
    "rook":   PieceSpec(58, 15, 17, 21, 26, 9.3),
    "pawn":   PieceSpec(50, 13, 12, 13, 20.5, 6.0),
}


# ---------------------------------------------------------------------------
# Bodies
# ---------------------------------------------------------------------------

def _densify(profile, step=1.5):
    """Insert points so no profile segment is longer than `step` mm."""
    p = np.asarray(profile, float)
    out = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]
    return np.array(out)


def _revolve(profile, sections=64) -> mesh_mod.Mesh:
    import trimesh

    tm = trimesh.creation.revolve(_densify(profile), sections=sections)
    return mesh_mod.Mesh(np.asarray(tm.vertices, float), np.asarray(tm.faces, np.int64))


def _sphere(r, center, subdiv=2) -> mesh_mod.Mesh:
    import trimesh

    tm = trimesh.creation.icosphere(subdivisions=subdiv, radius=r)
    return mesh_mod.Mesh(np.asarray(tm.vertices, float) + center, np.asarray(tm.faces, np.int64))


def _box(sx, sy, sz, center) -> mesh_mod.Mesh:
    cx, cy, cz = center
    return mesh_mod.box((sx, sy, sz), (cx - sx / 2, cy - sy / 2, cz - sz / 2))


def _base(R):
    return [(0, 0), (R, 0), (R, 3), (0.93 * R, 4.5), (0.93 * R, 6), (0.78 * R, 8)]


def body(kind: str) -> mesh_mod.Mesh:
    s = SPECS[kind]
    R, H = s.base_r, s.height
    parts = []
    if kind == "king":
        prof = _base(R) + [(0.55 * R, 14), (0.48 * R, 45), (0.52 * R, 55), (0.78 * R, 58),
                           (0.78 * R, 61), (0.5 * R, 63), (0.72 * R, 77), (0.35 * R, 80), (0, 80)]
        parts += [_box(3.6, 3.6, 12, (0, 0, 86)), _box(10, 3.6, 3.6, (0, 0, 87))]
    elif kind == "queen":
        prof = _base(R) + [(0.55 * R, 14), (0.47 * R, 42), (0.52 * R, 51), (0.78 * R, 54),
                           (0.78 * R, 57), (0.5 * R, 59), (0.82 * R, 73), (0.6 * R, 75), (0, 75)]
        for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
            parts.append(_sphere(1.8, np.array([0.7 * R * np.cos(a), 0.7 * R * np.sin(a), 75.5]), 1))
        parts.append(_sphere(3.5, np.array([0, 0, 79.5])))
    elif kind == "bishop":
        prof = _base(R) + [(0.55 * R, 13), (0.45 * R, 40), (0.72 * R, 44), (0.72 * R, 46),
                           (0.42 * R, 48), (0.56 * R, 56), (0.5 * R, 62), (0.25 * R, 67), (0, 68)]
        parts.append(_sphere(2.6, np.array([0, 0, 69.5])))
    elif kind == "rook":
        prof = _base(R) + [(0.62 * R, 12), (0.6 * R, 40), (0.8 * R, 43), (0.8 * R, 50), (0, 50)]
        for a in np.linspace(0, 2 * np.pi, 7)[:-1]:
            parts.append(_box(4.2, 4.2, 6, (0.66 * R * np.cos(a), 0.66 * R * np.sin(a), 53)))
    elif kind == "pawn":
        prof = _base(R) + [(0.5 * R, 13), (0.4 * R, 26), (0.66 * R, 28), (0.66 * R, 30),
                           (0.35 * R, 32), (0, 32)]
        parts.append(_sphere(0.56 * R, np.array([0, 0, 38.5]), 3))
    elif kind == "knight":
        prof = _base(R) + [(0.6 * R, 11), (0.6 * R, 13), (0, 13)]
        parts.append(_horse_head())
    else:
        raise ValueError(kind)
    _PROFILES[kind] = np.asarray(prof, float)
    return mesh_mod.combine([_revolve(prof)] + parts)


_PROFILES: dict[str, np.ndarray] = {}


def _horse_head(thickness=12.0, px=0.25) -> mesh_mod.Mesh:
    """Horse-head silhouette, puffed into a rounded slab (no extra libraries)."""
    from PIL import ImageDraw

    # Classic knight profile, snout pointing left (-x): neck, mane, ears, forehead,
    # snout, jaw, throat.
    pts = [(-7, 0), (9, 0), (10.5, 8), (11.5, 18), (11, 28), (8.5, 36), (5, 42),
           (4, 47), (2.2, 50), (0.8, 45.5), (-1.2, 49), (-3, 44), (-7, 40.5),
           (-12, 35), (-16, 29), (-17.8, 24.5), (-17.2, 20.5), (-14, 18.8), (-10, 19.8),
           (-6.5, 19), (-4.2, 15.5), (-5, 10), (-7, 4)]
    x0, y1 = -19.0, 52.0
    w, h = int(32 / px), int(54 / px)
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).polygon([((x - x0) / px, (y1 - y) / px) for x, y in pts], fill=255)
    m = np.asarray(im) > 127
    dome = relief.inflate(m)
    half = np.where(m, thickness / 2 * (0.55 + 0.45 * dome), 0)
    slab = mesh_mod.heightmap_to_mesh(half, m, px, bottom=-half)
    slab = slab.rotated_x(90)               # silhouette up -> z, thickness -> y
    lo, hi = slab.bounds()
    return slab.translated([-(lo[0] + hi[0]) / 2 + 1, -(lo[1] + hi[1]) / 2, 12 - lo[2]])


# ---------------------------------------------------------------------------
# Portrait cameo
# ---------------------------------------------------------------------------

@dataclass
class Face:
    img: Image.Image
    mask: np.ndarray
    depth: np.ndarray
    box: tuple  # face box inside img


def faces_from_photo(photo, log: list[str]) -> list[Face]:
    """Cut out the people and prepare one head-and-shoulders crop per face."""
    img = imaging.limit_size(imaging.load_image(photo), 1024)
    mask, eng = imaging.subject_mask(img)
    boxes = imaging.detect_faces(img, mask)
    boxes = sorted(boxes, key=lambda f: f[0])  # left to right
    log.append(f"Chess: {len(boxes)} face(s) found ({eng})")
    out = []
    for f in boxes:
        x0, y0, x1, y1 = imaging.portrait_crop_box(img.size, mask, [f], aspect=0.8)
        crop, m = img.crop((x0, y0, x1, y1)), mask[y0:y1, x0:x1].copy()
        fb = (f[0] - x0, f[1] - y0, f[2], f[3])
        # Keep only this person: every pixel closer (sideways) to another face
        # belongs to that other person (e.g. the partner in a couple photo).
        xx = np.arange(m.shape[1])[None, :] + x0
        cx = f[0] + f[2] / 2
        for g in boxes:
            if g is f:
                continue
            gx = g[0] + g[2] / 2
            mid = (cx + gx) / 2
            side = np.sign(gx - cx)
            m *= np.clip((mid - xx) * side / (0.08 * f[2]) + 0.5, 0, 1)
        m *= np.clip(1.6 - np.abs(xx - cx) / (f[2] * 1.1), 0, 1)
        d, _ = depth_mod.estimate_depth(crop, m, faces=[fb])
        out.append(Face(crop, m, d, fb))
    return out


def _oval(w, h, soft=1.5):
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - (w - 1) / 2) / (w / 2), (yy - (h - 1) / 2) / (h / 2))
    return r, np.clip((1 - r) * min(w, h) / (2 * soft), 0, 1)


WINDOW = 0.84   # the photo window / portrait area is 84% of the cameo size


def cameo(width_mm, height_mm, face: Face | None = None, initial: str = "",
          px: float = 0.15, style: str = "relief") -> mesh_mod.Mesh:
    """Oval medallion lying in x/y, front pointing to +z.

    style "relief": the face is sculpted in relief (single-colour print).
    style "photo" : a shallow oval window (1.2 mm deep) framed by a rim, for a
                    printed colour photo sticker + clear dome (see photo_sheet).
    """
    w, h = int(width_mm / px) | 1, int(height_mm / px) | 1
    r, inside = _oval(w, h)
    solid = r <= 1.0
    rim = np.clip((r - WINDOW) / 0.06, 0, 1) * (r <= 1.0)
    plate = 1.4
    height = np.full((h, w), plate)
    if style == "photo" and face is not None:
        floor = 1.2
        height = floor + rim * 1.2
        m = mesh_mod.heightmap_to_mesh(height, solid, px, bottom=0.0)
        lo, hi = m.bounds()
        return m.translated([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, 0])
    if face is not None:
        def fit(a):
            return np.asarray(Image.fromarray(np.asarray(a, np.float32)).resize((w, h), Image.BICUBIC),
                              dtype=np.float64)
        sx, sy = w / face.img.width, h / face.img.height
        fb = [(face.box[0] * sx, face.box[1] * sy, face.box[2] * sx, face.box[3] * sy)]
        detail = 0.3 + 0.7 * imaging.face_weight_map((h, w), fb)
        rel = relief.bas_relief(fit(face.depth), np.clip(fit(face.mask), 0, 1),
                                fit(imaging.to_gray(face.img)), compression=0.5, detail=detail)
        t = np.clip((np.arange(h) - h * 0.7) / (h * 0.3), 0, 1)       # fade shoulders
        rel *= (1 - t * t * (3 - 2 * t))[:, None]
        rel *= np.clip((WINDOW - r) / 0.08, 0, 1)
        height += rel * 1.6
    elif initial:
        tm = imaging.text_mask(initial, int(w * 0.6), int(h * 0.6))
        pad = np.zeros((h, w))
        y0, x0 = (h - tm.shape[0]) // 2, (w - tm.shape[1]) // 2
        pad[y0:y0 + tm.shape[0], x0:x0 + tm.shape[1]] = tm
        height += ndimage.gaussian_filter(pad, 1.0) * 1.2
    height = np.maximum(height, plate + rim * 1.0)
    m = mesh_mod.heightmap_to_mesh(height, solid, px, bottom=0.0)
    lo, hi = m.bounds()
    return m.translated([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, 0])


def piece(kind: str, face: Face | None = None, initial: str = "",
          style: str = "relief") -> mesh_mod.Mesh:
    """A complete piece standing on z=0, cameo facing the viewer (-y)."""
    s = SPECS[kind]
    b = body(kind)
    c = cameo(s.cameo_w, s.cameo_h, face, initial, style=style)
    c = c.rotated_x(90)  # relief (+z) now points to -y, image-up points to +z
    lo, hi = c.bounds()
    c = c.translated([0, -(front_depth(b, kind) - 1.0) - hi[1], s.cameo_z - (lo[2] + hi[2]) / 2])
    return mesh_mod.combine([b, c])


def front_depth(b: mesh_mod.Mesh, kind: str) -> float:
    """How far the body sticks out towards the viewer behind the cameo."""
    s = SPECS[kind]
    if kind == "knight" or kind not in _PROFILES:
        return s.body_r_at_cameo
    prof = _PROFILES[kind]
    zs = np.linspace(s.cameo_z - s.cameo_h / 2, s.cameo_z + s.cameo_h / 2, 60)
    # Walk the profile top-down from the base upwards; take the widest point.
    return float(np.interp(zs, prof[5:, 1], prof[5:, 0]).max())


# ---------------------------------------------------------------------------
# Colour photos for the "photo" style
# ---------------------------------------------------------------------------

def window_size_mm(kind: str) -> tuple[float, float]:
    s = SPECS[kind]
    return s.cameo_w * WINDOW, s.cameo_h * WINDOW


def window_photo(face: Face, w_mm: float, h_mm: float, dpi: int = 300,
                 background=((250, 238, 214), (214, 176, 120))) -> Image.Image:
    """Oval colour portrait exactly the size of a photo window (RGBA)."""
    W, H = int(w_mm / 25.4 * dpi), int(h_mm / 25.4 * dpi)
    img = face.img.convert("RGB").resize((W, H), Image.LANCZOS)
    m = np.asarray(Image.fromarray((np.clip(face.mask, 0, 1) * 255).astype(np.uint8))
                   .resize((W, H), Image.BILINEAR), dtype=np.float64) / 255.0
    m = ndimage.gaussian_filter(m, max(1.0, W / 40))[..., None]  # soft, feathered edges
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
    top, bot = np.array(background[0], float), np.array(background[1], float)
    bg = top[None, None] * (1 - rr[..., None] * 0.9) + bot[None, None] * rr[..., None] * 0.9
    rgb = np.asarray(img, float) * m + bg * (1 - m)
    alpha = (np.clip((1 - rr) * min(W, H) / 4, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), alpha]), "RGBA")


def photo_face_colors(m: mesh_mod.Mesh, kind: str, photo: Image.Image, base_rgb) -> np.ndarray:
    """Per-face colours: the photo inside the window, the material colour elsewhere."""
    s = SPECS[kind]
    ww, wh = window_size_mm(kind)
    tri = m.vertices[m.faces]
    cen = tri.mean(axis=1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    x, y, z = cen[:, 0], cen[:, 1], cen[:, 2]
    # Window floor sits in front of the body surface (y = -body_r); body faces don't.
    in_window = (n[:, 1] < -0.8) & (y < -(front_depth(body(kind), kind) + 0.05)) & \
        (((x / (ww / 2)) ** 2 + ((z - s.cameo_z) / (wh / 2)) ** 2) < 0.97)
    cols = np.tile(np.asarray(base_rgb, float), (len(cen), 1))
    ph = np.asarray(photo.convert("RGB"), float)
    ph_h, ph_w = ph.shape[:2]
    u = np.clip(((x + ww / 2) / ww * (ph_w - 1)).astype(int), 0, ph_w - 1)
    v = np.clip(((s.cameo_z + wh / 2 - z) / wh * (ph_h - 1)).astype(int), 0, ph_h - 1)
    cols[in_window] = ph[v[in_window], u[in_window]]
    return cols


def photo_sheet(photos: list[tuple[str, Image.Image]], dpi: int = 300) -> Image.Image:
    """A4 sheet of window photos at exact size, with cut lines and labels."""
    from PIL import ImageDraw

    from .imaging import get_font

    page = Image.new("RGB", (int(210 / 25.4 * dpi), int(297 / 25.4 * dpi)), "white")
    draw = ImageDraw.Draw(page)
    font = get_font(int(dpi * 0.1))
    draw.text((int(dpi * 0.4), int(dpi * 0.3)),
              "Family chess - photo windows. Print at 100% (actual size) on glossy "
              "sticker paper, cut along the grey line.", font=font, fill=(80, 80, 80))
    margin, gap = int(dpi * 0.4), int(dpi * 0.2)
    x, y, row_h = margin, int(dpi * 0.7), 0
    for label, ph in photos:
        w, h = ph.size
        if x + w > page.width - margin:
            x, y, row_h = margin, y + row_h + gap + int(dpi * 0.15), 0
        page.paste(ph, (x, y), ph)
        draw.ellipse([x - 2, y - 2, x + w + 2, y + h + 2], outline=(160, 160, 160), width=2)
        draw.text((x, y + h + 4), label, font=font, fill=(60, 60, 60))
        x += w + gap
        row_h = max(row_h, h)
    return page


# Standard back rank, left to right from white's side.
BACK_RANK = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"]
SQUARE_MM = 42.0
