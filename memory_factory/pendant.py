"""Face Pendant: a person's 3D head (Tripo model) becomes a raised portrait pendant for
lost-wax casting (jeweller's castable-resin print -> brass / silver), with a polished rim,
a cast-in loop for the chain and names + date engraved on the back.

Pipeline
    1. orient the head (face -> -y), find crown and chin automatically
    2. front depth map of the head -> bas-relief (soft, casting-safe, about 1.3 mm deep)
    3. frame (round / heart / oval), rim, loop, pupils
    4. back: engraved text (mirrored so it reads correctly) or a hollowed back (lighter)
    5. one closed heightmap solid -> STL for the jeweller + previews + notes + quote

Units: mm. See docs/face-pendant.md.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from . import costing, engrave, mesh as mesh_mod, orders, relief, render, royal
from .config import load_pricing

PX = 0.035                    # mm per heightmap pixel (fine enough for eyelids, bindi, beard)
SHAPES = ["Round", "Heart", "Oval"]
SIZES = {"Small 20 mm": 20.0, "Medium 24 mm": 24.0, "Large 28 mm": 28.0}
CROPS = ["Head & shoulders", "Face only"]
RIMS = ["Plain", "Beaded"]
BACKS = ["Engraved", "Hollow (lighter, no text)"]
DETAILS = ["Sharp", "Soft"]      # sharp: crisp features + outline step | soft: worn-coin look
# name -> (density g/cm3, preview material)
METALS = {
    "Gold-plated brass": (8.5, "gold"),
    "Antique gold brass": (8.5, "antique brass"),
    "Rhodium-plated brass (silver look)": (8.5, "silver"),
    "925 Silver": (10.4, "silver"),
}
FIELD = 1.0          # background plate thickness
RIM_H = 1.6          # rim height above the field
RELIEF = 1.3         # face relief depth at 24 mm (scales with size, capped)
ENGRAVE = 0.3        # back engraving depth
MIN_T = 0.6          # thinnest metal allowed anywhere (engraved letters under the field)
LOOP_R, LOOP_HOLE, LOOP_T = 2.4, 1.15, 1.8   # cast-in loop: outer radius, hole radius, thickness
HEART = "♥"


@dataclass
class PendantSettings:
    shape: str = "Round"
    size: str = "Medium 24 mm"
    crop: str = "Head & shoulders"
    rim: str = "Plain"
    back: str = "Engraved"
    metal: str = "Gold-plated brass"
    font: str = list(engrave.FONTS)[1]
    line1: str = ""                 # e.g. "Ram ♥ Meera"
    line2: str = ""                 # e.g. "Forever"
    date: str = ""
    pupils: bool = True
    detail: str = "Sharp"
    turn: str = "auto"
    packaging: str = "velvet box"


@dataclass
class Pendant:
    top: np.ndarray
    bottom: np.ndarray
    mask: np.ndarray
    info: dict = field(default_factory=dict)


@dataclass
class Result:
    order_id: str
    folder: Path
    files: list
    previews: list
    glb: Path | None
    quote_md: str
    log: list
    weights: dict


# ---------------------------------------------------------------- face -> relief
def chin_z(v: np.ndarray, cx: float) -> float:
    """Chin height: from the nose tip walk down the front profile until it drops away
    (under the chin / beard) by a third of the head depth."""
    z0, z1 = v[:, 2].min(), v[:, 2].max()
    h = z1 - z0
    top = v[v[:, 2] > z1 - 0.4 * h]
    c = v[np.abs(v[:, 0] - cx) < 0.12 * np.ptp(top[:, 0])]
    nb = 200
    idx = np.clip(((c[:, 2] - z0) / h * nb).astype(int), 0, nb - 1)
    f = np.full(nb, np.nan)
    order = np.argsort(c[:, 1])                  # front-most point per slice
    first = np.unique(idx[order], return_index=True)
    f[first[0]] = -c[order[first[1]], 1]
    ok = np.isfinite(f)                          # sparse models: bridge empty slices
    f = ndimage.gaussian_filter1d(np.interp(np.arange(nb), np.nonzero(ok)[0], f[ok]), 1.0)
    zc = z0 + (np.arange(nb) + 0.5) * h / nb
    i_nose = int(np.argmax(np.where(zc > z0 + 0.3 * h, f, -np.inf)))
    s = v[np.abs(v[:, 2] - zc[i_nose]) < 0.01 * h]
    depth = np.ptp(s[:, 1])
    i = i_nose
    while i > 0 and f[i - 1] > f[i_nose] - 0.3 * depth:
        i -= 1
    return float(zc[i])


def face_depth(head_path, turn: str, face_mm: float, below_mm: float, width_mm: float,
               log: list[str]):
    """Front depth map (rows from the crown down) with crown-to-chin = face_mm."""
    import trimesh

    head_path = Path(head_path)
    m = royal.load_head(head_path)
    v = royal.orient(m, head_path.suffix, turn, log)
    lm = royal.landmarks(v, None)
    zc = chin_z(v, lm["cx"])
    z1 = v[:, 2].max()
    chin_frac = (z1 - zc) / max(np.ptp(v[:, 2]), 1e-6)
    k = face_mm / max(z1 - zc, 1e-6)
    v = (v - [lm["cx"], 0, z1]) * k
    pts = trimesh.Trimesh(v, m.faces, process=False).sample(7_000_000)
    rows = int(round((face_mm + below_mm) / PX))
    half = int(round(width_mm / 2 / PX))
    i = np.floor(-pts[:, 2] / PX).astype(int)
    j = np.floor(pts[:, 0] / PX).astype(int) + half
    ok = (i >= 0) & (i < rows) & (j >= 0) & (j < 2 * half)
    d = np.full((rows, 2 * half), -np.inf)
    np.maximum.at(d, (i[ok], j[ok]), -pts[ok, 1])
    have = np.isfinite(d)
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(have, iterations=3))
    idx = ndimage.distance_transform_edt(~have, return_distances=False, return_indices=True)
    d = ndimage.median_filter(d[tuple(idx)], 3)
    log.append(f"Face: crown-to-chin {face_mm:.1f} mm (chin found {chin_frac:.0%} down the model)")
    return d, mask


STEP = 0.15         # crisp outline step around the head (sharp detail), mm


def bust_relief(d, mask, depth_mm: float, fade_mm: float, sharp: bool = True):
    """Depth map -> relief heights in mm. Sharp keeps more of the fine shape (eyelids, lips,
    beard, hair strands), slightly exaggerated because casting and polishing soften it, and
    gives the head a clean outline step like a coin portrait. Soft is the worn-coin look."""
    mask = ndimage.gaussian_filter(mask.astype(float), 0.06 / PX) > 0.5     # smooth outline
    below = np.cumsum(mask[::-1], axis=0)[::-1]           # mask rows under each pixel
    fade = np.clip(below * PX / fade_mm, 0, 1)
    fade = fade * fade * (3 - 2 * fade)
    if not sharp:
        r = relief.bas_relief(d, mask, compression=0.35, detail=0.6, edge_softness_px=0.4 / PX)
        return ndimage.gaussian_filter(r * fade, 0.05 / PX) * depth_mm
    r = relief.bas_relief(d, mask, compression=0.55, detail=1.2, edge_softness_px=0.15 / PX)
    hp = d - ndimage.gaussian_filter(d, 0.3 / PX)         # fine shape, exaggerated a little
    hp = np.clip(hp / (np.percentile(np.abs(hp[mask]), 99) + 1e-9), -1, 1) * mask
    edge = np.clip(ndimage.distance_transform_edt(mask) * PX / 0.07, 0, 1)
    h = (r + 0.12 * hp) * depth_mm + STEP * edge
    return ndimage.gaussian_filter(h * fade, 0.014 / PX)


def find_pupils(d, mask, face_mm: float):
    """Eye centres in the depth map (rows, cols) or None when not confident."""
    rows = int(face_mm / PX)
    band = slice(int(0.36 * rows), int(0.58 * rows))
    hp = d - ndimage.gaussian_filter(d, 0.1 * face_mm / PX)
    hp = ndimage.gaussian_filter(hp, 0.025 * face_mm / PX)
    cols = np.nonzero(mask[int(0.45 * rows)])[0]
    if len(cols) < 10:
        return None
    mid = (cols[0] + cols[-1]) / 2
    wid = cols[-1] - cols[0]
    found = []
    for sgn in (-1, 1):
        lo, hi = sorted((mid + sgn * 0.08 * wid, mid + sgn * 0.36 * wid))
        sub = hp[band, int(lo):int(hi)]
        if sub.size == 0:
            return None
        r, c = np.unravel_index(np.argmax(sub), sub.shape)
        found.append((band.start + r, int(lo) + c))
    (r1, c1), (r2, c2) = found
    if abs(r1 - r2) > 0.06 * rows or abs(abs(c1 - mid) - abs(c2 - mid)) > 0.12 * wid:
        return None
    return found


# ---------------------------------------------------------------- frame
def frame(shape: str, size: float):
    """Body mask of the pendant (no loop) on a canvas with room for the loop on top.
    Returns mask, inside-distance (mm), canvas origin info."""
    if shape == "Round":
        W, H = size, size
    elif shape == "Oval":
        W, H = size * 0.92, size * 1.25          # Medium: 22 x 30 mm
    elif shape == "Heart":
        W, H = size, size * 0.92
    else:
        raise ValueError(f"Unknown shape {shape}")
    pad_top = 2 * LOOP_R + 0.5
    nx = int(round((W + 1.0) / PX)) + 1
    ny = int(round((H + 1.0 + pad_top) / PX)) + 1
    yy, xx = np.mgrid[0:ny, 0:nx] * PX
    x = xx - (W + 1.0) / 2
    y = yy - pad_top - 0.5                              # y down, 0 = top of the body box
    if shape == "Round":
        body = np.hypot(x, y - H / 2) <= W / 2
    elif shape == "Oval":
        body = np.hypot(x / (W / 2), (y - H / 2) / (H / 2)) <= 1
    else:
        # classic implicit heart; its exact extent: |X| <= 1.1394, -1 <= Y <= 1.2356
        X = x / (W / 2) * 1.1394
        Y = 1.2356 - y / H * 2.2356
        body = (X ** 2 + Y ** 2 - 1) ** 3 - X ** 2 * Y ** 3 <= 0
    dist = ndimage.distance_transform_edt(body) * PX
    return body, dist, x, y, W, H


def rim_profile(dist, x, y, H, rim_w, beaded: bool):
    u = np.clip(1 - ((dist - rim_w / 2) / (rim_w / 2)) ** 2, 0, 1) ** 0.5
    h = RIM_H * u
    if beaded:
        ang = np.arctan2(y - H / 2, x)
        n = int(np.pi * H / (rim_w * 0.9))
        h = h * (0.75 + 0.25 * (0.5 + 0.5 * np.cos(ang * n)) ** 0.6)
    return np.where(dist <= rim_w, h, 0.0)


def text_line(text: str, font_file: str, height_mm: float) -> np.ndarray:
    """Bool image of one line of text at PX resolution, ink height = height_mm.
    '♥' is drawn as a vector heart (the bundled fonts have no heart glyph)."""
    f = ImageFont.truetype(str(engrave.FONT_DIR / font_file), 200)
    cap = f.getbbox("M")
    cap_h = cap[3] - cap[1]
    parts = text.split(HEART)
    widths = [f.getbbox(p)[2] - f.getbbox(p)[0] if p.strip() else int(f.getlength(p)) for p in parts]
    hw = int(cap_h * 0.95)
    gap = int(cap_h * 0.25)
    W = sum(widths) + (len(parts) - 1) * (hw + 2 * gap) + 80
    im = Image.new("L", (W, 400), 0)
    d = ImageDraw.Draw(im)
    xpos = 40
    for i, p in enumerate(parts):
        if p:
            bb = f.getbbox(p)
            d.text((xpos - bb[0], 100), p, font=f, fill=255)
        xpos += widths[i]
        if i < len(parts) - 1:
            t = np.linspace(0, 2 * np.pi, 120)
            hx = 16 * np.sin(t) ** 3
            hy = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
            s = hw / 34
            y0 = 100 + cap[1] + cap_h * 0.5
            d.polygon([(xpos + gap + hw / 2 + a * s, y0 - b * s) for a, b in zip(hx, hy)], fill=255)
            xpos += hw + 2 * gap
    a = np.asarray(im) > 127
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        raise ValueError(f'"{text}" has no printable letters in this font.')
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    k = height_mm / PX / a.shape[0]
    out = np.asarray(Image.fromarray(a.astype(np.uint8) * 255).resize(
        (max(1, int(a.shape[1] * k)), max(1, int(round(height_mm / PX)))), Image.LANCZOS)) > 110
    return ndimage.binary_dilation(out, iterations=1)


def fit_back_text(s: PendantSettings, inside: np.ndarray, size: float):
    """Lay the back text out in the biggest circle inside the pendant. Returns a bool
    mask (as seen from the BACK) or None."""
    lines = []
    font_file, kind = engrave.FONTS[s.font]
    date_file, _ = engrave.DATE_FONT
    for t, ff, rel, kd in ((s.line1, font_file, 0.14, kind), (s.line2, font_file, 0.10, kind),
                           (s.date, date_file, 0.075, "date")):
        if t.strip():
            lines.append((t.strip(), ff, rel, kd))
    if not lines:
        return None
    mins = {"script": 2.0, "caps": 1.4, "italic": 1.7, "date": 1.2}   # cast metal stays crisp small
    ins = ndimage.binary_erosion(inside, iterations=int(0.5 / PX))
    rows_in = np.nonzero(ins.any(1))[0]
    cols_in = np.nonzero(ins.any(0))[0]
    cx = (cols_in[0] + cols_in[-1]) // 2
    cy = int(ndimage.center_of_mass(ins)[0])

    def span(r0, r1):          # free width around cx shared by all rows r0..r1
        if r0 < rows_in[0] or r1 > rows_in[-1]:
            return 0
        free = ins[r0:r1 + 1].all(0)
        blocked = np.nonzero(~free)[0]
        left = blocked[blocked <= cx]
        right = blocked[blocked >= cx]
        if (len(left) and left[-1] == cx) or (len(right) and right[0] == cx):
            return 0
        half_l = cx - (left[-1] if len(left) else -1)
        half_r = (right[0] if len(right) else len(free)) - cx
        return 2 * min(half_l, half_r) - 1

    scale = [1.0] * len(lines)              # each line shrinks on its own until it fits
    for _ in range(80):
        imgs = [text_line(t, ff, rel * size * k) for (t, ff, rel, _), k in zip(lines, scale)]
        gap = int(0.35 * min(i.shape[0] for i in imgs))
        total = sum(i.shape[0] for i in imgs) + gap * (len(imgs) - 1)
        y = int(cy - total / 2)
        bad = []
        for n, im in enumerate(imgs):
            if im.shape[1] + 2 > span(y, y + im.shape[0]):
                bad.append(n)
            y += im.shape[0] + gap
        if not bad:
            break
        for n in bad:
            scale[n] *= 0.95
            t, ff, rel, kd = lines[n]
            if rel * size * scale[n] < mins[kd]:
                raise ValueError(f'Back text "{t}" is too long for this pendant size - shorten it '
                                 "or choose a bigger size.")
    out = np.zeros(inside.shape, bool)
    y = int(cy - total / 2)
    for im in imgs:
        x0 = int(cx - im.shape[1] / 2)
        out[y:y + im.shape[0], x0:x0 + im.shape[1]] |= im
        y += im.shape[0] + gap
    return out


def check_text(s: PendantSettings) -> None:
    """Fail fast (before any Tripo credits are spent) when the back text cannot fit."""
    if s.back != BACKS[0]:
        return
    size = SIZES[s.size]
    body, dist, *_ = frame(s.shape, size)
    fit_back_text(s, body & (dist > 1.1 * size / 24 + 0.3), size)


# ---------------------------------------------------------------- build
def build(head_path, s: PendantSettings, log: list[str]) -> Pendant:
    size = SIZES[s.size]
    body, dist, x, y, W, H = frame(s.shape, size)
    k = size / 24.0
    rim_w = 1.1 * k
    depth = min(RELIEF * k, 1.45)
    inner = dist > rim_w + 0.3
    # face placement inside the rim
    cols = np.nonzero(body.any(0))[0]
    cxi = (cols[0] + cols[-1]) // 2
    top_row = np.nonzero(inner[:, cxi])[0][0]
    bot_row = np.nonzero(inner[:, cxi])[0][-1]
    Hi = (bot_row - top_row) * PX
    if s.crop == CROPS[0]:
        face_mm, below, gap = 0.60 * Hi, 0.40 * Hi, 0.06 * Hi
    else:
        face_mm, below, gap = 0.76 * Hi, 0.14 * Hi, 0.06 * Hi
    if s.shape == "Heart":                       # the face sits in the wide upper part
        below *= 0.7
    d, fmask = face_depth(head_path, s.turn, face_mm, below, W, log)
    rel = bust_relief(d, fmask, depth, fade_mm=max(1.5, 0.12 * Hi), sharp=s.detail == DETAILS[0])
    if s.pupils:
        eyes = find_pupils(d, fmask, face_mm)
        if eyes:
            rr, cc = np.mgrid[0:rel.shape[0], 0:rel.shape[1]]
            rad = 0.024 * face_mm / PX
            for (r0, c0) in eyes:
                q = np.hypot(rr - r0, cc - c0) / rad
                rel -= np.where(q < 1, 0.5 + 0.5 * np.cos(np.pi * q), 0) * 0.09 * k
            log.append("Pupils: tiny dimples added to the eyes (look more alive in metal)")
        else:
            log.append("Pupils: eyes not found confidently - left plain")
    r0 = top_row + int(gap / PX)
    c0 = cxi - rel.shape[1] // 2
    face = np.zeros(body.shape)
    h_ = min(rel.shape[0], face.shape[0] - r0)
    face[r0:r0 + h_, c0:c0 + rel.shape[1]] = rel[:h_]
    face *= np.clip((dist - rim_w - 0.35) / 0.35, 0, 1)
    rim = rim_profile(dist, x, y, H, rim_w, s.rim == "Beaded")
    top = np.where(body, FIELD + np.maximum(face, rim), 0.0)
    # cast-in loop above the top edge (between the lobes for a heart)
    rows = np.nonzero(body[:, cxi])[0]
    ry, rx = rows[0] * PX - (LOOP_R - 0.9), cxi * PX
    rr = np.hypot(np.arange(body.shape[1])[None, :] * PX - rx, np.arange(body.shape[0])[:, None] * PX - ry)
    ring = (rr <= LOOP_R) & (rr >= LOOP_HOLE)
    hw = (LOOP_R - LOOP_HOLE) / 2
    ring_h = 1.0 + (LOOP_T - 1.0) * np.sqrt(np.clip(1 - ((rr - LOOP_HOLE - hw) / hw) ** 2, 0, 1))
    mask = body | ring
    top = np.where(ring & ~body, ring_h, top)
    top = np.where(ring & body, np.maximum(top, ring_h), top)
    idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    top = np.where(mask, ndimage.gaussian_filter(top[tuple(idx)], 0.7), 0)   # blur without thinning edges
    # back
    bottom = np.zeros_like(top)
    text = None
    if s.back == BACKS[0]:
        text = fit_back_text(s, inner & body, size)
        if text is not None:
            back_view = text[:, ::-1]              # mirrored: reads correctly from the back
            bottom = ndimage.gaussian_filter(back_view.astype(float), 0.6) * ENGRAVE
    else:
        hollow = np.clip(face - 0.15, 0, None) * np.clip((dist - rim_w - 0.8) / 0.5, 0, 1)
        bottom = ndimage.gaussian_filter(hollow, 2.0)
    bottom = np.where(mask, np.minimum(bottom, top - MIN_T), 0)
    bottom = np.maximum(bottom, 0)
    thick = np.where(mask, top - bottom, np.inf)
    info = dict(size_mm=[round(W, 1), round(H + 2 * LOOP_R - 0.9, 1)], relief_mm=round(float(depth), 2),
                min_thickness_mm=round(float(thick.min()), 2), loop_hole_mm=round(2 * LOOP_HOLE, 1),
                face_mm=round(face_mm, 1), back_text=bool(text is not None and text.any()))
    return Pendant(top, bottom, mask, info)


def to_trimesh(p: Pendant):
    import trimesh

    m = mesh_mod.heightmap_to_mesh(p.top, p.mask, pixel_mm=PX, bottom=p.bottom)
    return trimesh.Trimesh(m.vertices, m.faces, process=False)


def finish_mesh(t, log):
    t = royal.simplify(t, 450_000, 0.004)
    t = royal.weld_short_edges(t, 0.01, log)
    return t


def weights(volume_mm3: float) -> dict:
    return {name: round(volume_mm3 / 1000 * dens, 1) for name, (dens, _) in METALS.items()}


def _antique(img: Image.Image, height: np.ndarray, mask: np.ndarray, low: float,
             span: float = 0.35) -> Image.Image:
    """Antique / oxidised look: recesses and the background darkened, raised parts polished."""
    t = np.clip((height - low) / span, 0, 1)
    t = ndimage.gaussian_filter(t, 0.03 / PX)
    k = np.where(mask, 0.42 + 0.58 * t, 1.0)[..., None]
    return Image.fromarray(np.clip(np.asarray(img, float) * k, 0, 255).astype(np.uint8))


def preview(p: Pendant, s: PendantSettings) -> tuple[Image.Image, Image.Image]:
    mat = METALS[s.metal][1]
    front = render.render_material(p.top, PX, p.mask, mat)
    back_h = -p.bottom[:, ::-1]
    back = render.render_material(back_h, PX, p.mask[:, ::-1], mat)
    if "Antique" in s.metal:
        front = _antique(front, p.top, p.mask, FIELD + 0.02)
        back = _antique(back, back_h, p.mask[:, ::-1], -ENGRAVE, 0.6 * ENGRAVE)   # dark letters
    return front, back


def jeweller_notes(s: PendantSettings, p: Pendant, wts: dict, label: str) -> str:
    i = p.info
    lines = [
        f"Face Pendant - {label}",
        "=" * 40,
        f"Shape: {s.shape}   Size: {i['size_mm'][0]} x {i['size_mm'][1]} mm (incl. loop)",
        f"Metal / finish: {s.metal}",
        f"Approx. weight: brass {wts['Gold-plated brass']} g, 925 silver {wts['925 Silver']} g",
        "",
        "Process: castable-resin print (25-50 micron) -> lost-wax casting -> polish.",
        "File units: millimetres. Do not scale (add your usual casting shrinkage allowance).",
        f"Face relief {i['relief_mm']} mm; rim {RIM_H} mm above a {FIELD} mm plate; "
        f"thinnest metal {i['min_thickness_mm']} mm.",
        f"Cast-in loop with a {i['loop_hole_mm']} mm hole: add a jump ring for the chain.",
        "Sprue: on the back edge, NOT on the face or the loop.",
    ]
    if i["back_text"]:
        lines.append(f"Back: engraved text {ENGRAVE} mm deep - keep it crisp when polishing.")
    else:
        lines.append("Back: hollowed behind the face to save metal.")
    if "Gold" in s.metal or "Rhodium" in s.metal or "Antique" in s.metal:
        lines += ["", "Plating: NICKEL-FREE, gold 2-3 micron or more, then clear lacquer (anti-tarnish)."]
    if "Antique" in s.metal:
        lines.append("Antique finish: darken the background, polish the face and rim bright.")
    if "Silver" in s.metal:
        lines += ["", "925 silver: stamp '925' on the back; optional oxidised background."]
    lines += ["", "Polish lightly on the face: the nose and cheeks are the high points."]
    return "\n".join(lines) + "\n"


def generate(sources: list[tuple[str, str]], s: PendantSettings, customer: dict | None = None,
             api_key: str | None = None, progress=None) -> Result:
    """sources: [(path, label)] - one pendant per head model (or photo -> Tripo)."""
    from .providers import tripo

    customer = customer or {}
    log: list[str] = []
    t_all = time.time()
    check_text(s)
    order_id, folder = orders.new_order_dir("pendant", customer.get("name", ""))
    files, previews, total_vol, all_w, glb = [], [], 0.0, {}, None
    for n, (src, label) in enumerate(sources):
        src = Path(src)
        tag = "".join(c for c in label.lower() if c.isalnum()) or f"p{n + 1}"
        if progress:
            progress(0.05 + 0.9 * n / len(sources), desc=f"Pendant {n + 1}: face relief...")
        if src.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            head = tripo.photo_to_head_model(src, api_key or "", folder, log, progress=progress)
        else:
            head = folder / f"head_{tag}{src.suffix.lower()}"
            head.write_bytes(src.read_bytes())
        log.append(f"--- Pendant {n + 1} ({label}) ---")
        p = build(head, s, log)
        t = finish_mesh(to_trimesh(p), log)
        if not t.is_watertight:
            log.append("WARNING: mesh is not watertight - check before sending to the jeweller")
        vol = float(abs(t.volume))
        total_vol += vol
        w = weights(vol)
        all_w[label] = w
        stl = folder / f"{order_id}_{tag}.stl"
        t.export(stl)
        front, back = preview(p, s)
        both = Image.new("RGB", (front.width * 2 + 40, front.height), (245, 242, 236))
        both.paste(front, (0, 0))
        both.paste(back, (front.width + 40, 0))
        img = folder / f"{order_id}_{tag}_front_back.png"
        both.resize((both.width // 2, both.height // 2), Image.LANCZOS).save(img)
        notes = folder / f"{order_id}_{tag}_jeweller_notes.txt"
        notes.write_text(jeweller_notes(s, p, w, label), encoding="utf-8")
        files += [stl, notes, img]
        previews.append(img)
        log.append(f"Size {p.info['size_mm'][0]} x {p.info['size_mm'][1]} mm, thinnest metal "
                   f"{p.info['min_thickness_mm']} mm, weight ≈ {w[s.metal]} g in {s.metal}")
        if glb is None:
            step = 5
            small = mesh_mod.heightmap_to_mesh(p.top[::step, ::step], p.mask[::step, ::step],
                                               PX * step, bottom=p.bottom[::step, ::step])
            glb = mesh_mod.write_glb_preview(small, folder / "preview.glb",
                                             color=render.MATERIALS[METALS[s.metal][1]][1])
    q = costing.quote_pendant(total_vol / len(sources), s.metal, len(sources), s.packaging)
    gst = load_pricing()["gst_percent"]
    orders.save_order(folder, {"order_id": order_id, "product": "face pendant", "customer": customer,
                               "quantity": len(sources), "settings": asdict(s),
                               "labels": [lbl for _, lbl in sources], "weights_g": all_w,
                               "quote": {"price_per_piece": q.price_per_piece, "total_price": q.total_price,
                                         "lines": q.lines},
                               "status": "proof sent", "log": log})
    log.append(f"Total time {time.time() - t_all:.0f}s")
    return Result(order_id, folder, files, previews, glb, q.as_markdown(gst), log, all_w)
