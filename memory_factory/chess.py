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
    "pawn":   PieceSpec(50, 13, 14, 17, 22, 6.0),
}


# ---------------------------------------------------------------------------
# Bodies
# ---------------------------------------------------------------------------

def _revolve(profile, sections=64) -> mesh_mod.Mesh:
    import trimesh

    tm = trimesh.creation.revolve(np.asarray(profile, float), sections=sections)
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
    return mesh_mod.combine([_revolve(prof)] + parts)


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


def cameo(width_mm, height_mm, face: Face | None = None, initial: str = "",
          px: float = 0.15) -> mesh_mod.Mesh:
    """Oval medallion lying in x/y with the relief pointing to +z."""
    w, h = int(width_mm / px) | 1, int(height_mm / px) | 1
    r, inside = _oval(w, h)
    solid = r <= 1.0
    rim = np.clip((r - 0.84) / 0.06, 0, 1) * (r <= 1.0)
    plate = 1.4
    height = np.full((h, w), plate)
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
        rel *= np.clip((0.84 - r) / 0.08, 0, 1)
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


def piece(kind: str, face: Face | None = None, initial: str = "") -> mesh_mod.Mesh:
    """A complete piece standing on z=0, cameo facing the viewer (-y)."""
    s = SPECS[kind]
    c = cameo(s.cameo_w, s.cameo_h, face, initial)
    c = c.rotated_x(90)  # relief (+z) now points to -y, image-up points to +z
    lo, hi = c.bounds()
    c = c.translated([0, -(s.body_r_at_cameo - 1.0) - hi[1], s.cameo_z - (lo[2] + hi[2]) / 2])
    return mesh_mod.combine([body(kind), c])


# Standard back rank, left to right from white's side.
BACK_RANK = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"]
SQUARE_MM = 42.0
