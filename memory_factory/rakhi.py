"""Face Rakhi: the brother's 3D face (Tripo head model) in a rakhi, plus an optional matching
lumba with his wife's (bhabhi's) face.

Each piece is a decorative top with:
  - the raised face (same relief engine as the Face Pendant)
  - recessed seats for glue-in flat-back stones (kundan / rhinestones, 3 mm and 2 mm)
  - a thread tunnel across the back (rakhi; the dori slides through, no sewing / glue)
  - a small loop on top: after the festival the thread slides out and it becomes a keychain
    or pendant (the lumba hangs from this loop)

Two ways to make it:
  Resin (painted)  SLA print, painted gold - sharpest face (sell ~Rs 499-599)
  FDM (silk gold)  silk gold PLA, no painting - 20% bigger, deeper relief (sell ~Rs 299-399)

Units: mm. See docs/face-rakhi.md.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from . import costing, engrave, mesh as mesh_mod, orders, pendant, render, royal
from .config import load_pricing

PX = pendant.PX
DESIGNS = ["Kundan medallion", "Flower"]
SETS = ["Rakhi only", "Bhaiya + Bhabhi set (rakhi + lumba)"]
METHODS = {
    # name -> (outline scale, face relief mm, raised text mm, smallest letter mm)
    "Resin (painted gold)": (1.0, 1.2, 0.35, 1.6),
    "FDM (silk gold PLA)": (1.2, 2.0, 0.6, 2.5),
}
STONES = {
    "Red & green": [(200, 25, 45), (20, 130, 70)],
    "Red": [(200, 25, 45)],
    "Blue & white": [(30, 70, 190), (245, 245, 240)],
    "Pearl white": [(245, 240, 230)],
}
FIELD = 1.0
SEAT_DEPTH = 0.6                 # stone seats: flat-bottom cups
STONE_BIG, STONE_SMALL = 3.0, 2.0
BOSS = 3.6                       # back ridge that carries the thread tunnel
TUNNEL_R = 1.0                   # 2 mm tunnel: fits a rakhi dori / 1.5 mm cord
LOOP_R, LOOP_HOLE, LOOP_T = 2.4, 1.15, 1.8


@dataclass
class RakhiSettings:
    design: str = DESIGNS[0]
    set_type: str = SETS[0]
    method: str = list(METHODS)[0]
    name: str = "BHAIYA"             # on the kundan medallion, under the face
    font: str = list(engrave.FONTS)[1]
    stones: str = "Red & green"
    face_scale: float = 1.0
    face_shift: float = 0.0          # mm, + moves the face down
    turn: str = "auto"
    packaging: str = "gift box"


@dataclass
class Piece:
    kind: str                        # "rakhi" | "lumba"
    top: np.ndarray
    bottom: np.ndarray
    mask: np.ndarray
    seats: list                      # (row, col, radius_mm)
    tunnel: tuple | None             # (row, col0, col1) in pixels
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


def _grid(size: float):
    pad = 2 * LOOP_R + 0.5
    n = int(round((size + 1.0) / PX)) + 1
    ny = int(round((size + 1.0 + pad) / PX)) + 1
    yy, xx = np.mgrid[0:ny, 0:n] * PX
    x = xx - (size + 1.0) / 2
    y = yy - pad - 0.5 - size / 2            # 0 at the centre of the round body
    return x, y


def _dome(r, R, h):
    return h * np.sqrt(np.clip(1 - (r / R) ** 2, 0, 1))


def _place(shape, rel, cy_mm, y0_px):
    """Paste rel so that its centre sits cy_mm below the body centre (row y0_px)."""
    out = np.zeros(shape)
    H, W = shape
    r0 = int(y0_px + cy_mm / PX - rel.shape[0] / 2)
    c0 = W // 2 - rel.shape[1] // 2
    ys, ye = max(r0, 0), min(r0 + rel.shape[0], H)
    xs, xe = max(c0, 0), min(c0 + rel.shape[1], W)
    if ye > ys and xe > xs:
        out[ys:ye, xs:xe] = rel[ys - r0:ye - r0, xs - c0:xe - c0]
    return out


def _face(head, s: RakhiSettings, face_mm, below, width, depth, log):
    sc = float(np.clip(s.face_scale, 0.6, 1.5))
    d, m = pendant.face_depth(head, s.turn, face_mm * sc, below * sc, width, log)
    rel = pendant.bust_relief(d, m, depth, fade_mm=1.5, sharp=True)
    eyes = pendant.find_pupils(d, m, face_mm * sc)
    if eyes:
        rr, cc = np.mgrid[0:rel.shape[0], 0:rel.shape[1]]
        for (r0, c0) in eyes:
            q = np.hypot(rr - r0, cc - c0) / (0.024 * face_mm * sc / PX)
            rel -= np.where(q < 1, 0.5 + 0.5 * np.cos(np.pi * q), 0) * 0.08
    return rel


def _seats(top, x, y, centres, stone_mm, pad_h):
    """Raised bezel pads with flat-bottom cups for glue-in flat-back stones of stone_mm."""
    seats = []
    cup = stone_mm / 2 + 0.05                    # a little play for glue
    for cx, cy in centres:
        r = np.hypot(x - cx, y - cy)
        pad = np.where(r <= cup + 0.55, FIELD + pad_h - 0.4 * np.clip((r - cup) / 0.55, 0, 1), 0)
        top = np.maximum(top, pad)
        top = np.where(r <= cup, FIELD + pad_h - SEAT_DEPTH, top)
        row = int(np.argmin(np.abs(y[:, 0] - cy)))
        col = int(np.argmin(np.abs(x[0] - cx)))
        seats.append((row, col, stone_mm / 2))
    return top, seats


def _loop(top, mask, x, y, y_top):
    r = np.hypot(x, y - (y_top - (LOOP_R - 0.9)))
    ring = (r <= LOOP_R) & (r >= LOOP_HOLE)
    hw = (LOOP_R - LOOP_HOLE) / 2
    h = 1.0 + (LOOP_T - 1.0) * np.sqrt(np.clip(1 - ((r - LOOP_HOLE - hw) / hw) ** 2, 0, 1))
    top = np.where(ring, np.maximum(top, h), top)
    return top, mask | ring


def check_text(s: RakhiSettings):
    """The name under the face (kundan medallion). Fails fast, before any Tripo credits are spent.
    Returns the text image or None."""
    if s.design != DESIGNS[0] or not s.name.strip():
        return None
    k, _, _, min_letter = METHODS[s.method]
    Ri = 16.0 * k - 4.9 * k
    t = pendant.text_line(s.name.strip(), engrave.FONTS[s.font][0], max(min_letter, 0.17 * Ri))
    if t.shape[1] * PX > 1.35 * Ri:
        raise ValueError(f'Name "{s.name}" is too long for the rakhi - use up to about '
                         f"{max(3, int(len(s.name) * 1.35 * Ri / (t.shape[1] * PX)))} letters.")
    return t


def build_rakhi(head, s: RakhiSettings, log: list[str]) -> Piece:
    k, depth, text_h, min_letter = METHODS[s.method]
    size = (32.0 if s.design == DESIGNS[0] else 40.0) * k
    x, y = _grid(size)
    r = np.hypot(x, y)
    a = np.arctan2(y, x)
    y0 = int(np.argmin(np.abs(y[:, 0])))
    top = np.full(r.shape, FIELD)
    if s.design == DESIGNS[0]:                               # kundan medallion
        R = size / 2
        body = r <= R
        top = np.maximum(top, FIELD + _dome(np.abs(r - (R - 1.0 * k)), 1.0 * k, 1.2))
        centres = [((R - 3.1 * k) * np.cos(t), (R - 3.1 * k) * np.sin(t))
                   for t in np.arange(16) * np.pi / 8]
        top, seats = _seats(top, x, y, centres, STONE_BIG, 1.1)
        Ri = R - 4.9 * k
        face = _face(head, s, 0.62 * 2 * Ri, 0.16 * 2 * Ri, 2 * Ri, depth, log)
        top = top + _place(r.shape, face, -0.17 * Ri + s.face_shift, y0) * np.clip((Ri - 0.3 - r) / 0.5, 0, 1)
        t = check_text(s)
        if t is not None:
            tz = _place(r.shape, ndimage.gaussian_filter(t.astype(float), 1.0), 0.74 * Ri, y0)
            top = top + tz * text_h
    else:                                                    # flower
        Rc = 0.285 * size
        petal = 0.375 * size + 0.105 * size * np.abs(np.cos(6 * a)) ** 0.6
        body = r <= petal
        top = FIELD + 0.9 * np.clip((petal - r) / (2.5 * k), 0, 1) * (0.6 + 0.4 * np.cos(12 * a) ** 2)
        top = np.maximum(top, FIELD + _dome(np.abs(r - (Rc + 0.8 * k)), 0.8 * k, 1.4))
        centres = [(0.405 * size * np.cos(t), 0.405 * size * np.sin(t)) for t in np.arange(12) * np.pi / 6]
        top, seats = _seats(top, x, y, centres, STONE_BIG, 1.3)
        top = np.where(r < Rc, FIELD, top)
        face = _face(head, s, 0.68 * 2 * Rc, 0.18 * 2 * Rc, 2 * Rc, depth, log)
        top = top + _place(r.shape, face, -0.06 * Rc + s.face_shift, y0) * np.clip((Rc - 0.4 - r) / 0.5, 0, 1)
    top = np.where(body, top, 0)
    y_top = y[np.nonzero(body[:, body.shape[1] // 2])[0][0], 0]
    top, mask = _loop(top, body, x, y, y_top)
    # back: a ridge across the middle carrying the thread tunnel
    cols = np.nonzero(body[y0])[0]
    c0, c1 = cols[0] + int(3.0 / PX), cols[-1] - int(3.0 / PX)
    band = np.clip((2.4 * k - np.abs(y)) / 0.6, 0, 1) * np.clip(
        (np.minimum(np.arange(r.shape[1]) - c0, c1 - np.arange(r.shape[1]))[None, :] * PX) / 0.6, 0, 1)
    bottom = -BOSS * np.sqrt(band) * body
    return _finish(Piece("rakhi", top, bottom, mask, seats, (y0, c0, c1)), size, y0)


def build_lumba(head, s: RakhiSettings, log: list[str]) -> Piece:
    k, depth, _, _ = METHODS[s.method]
    size = 22.0 * k
    x, y = _grid(size)
    r = np.hypot(x, y)
    y0 = int(np.argmin(np.abs(y[:, 0])))
    R = size / 2
    body = r <= R
    top = np.maximum(np.full(r.shape, FIELD), FIELD + _dome(np.abs(r - (R - 0.8 * k)), 0.8 * k, 1.1))
    centres = [((R - 2.4 * k) * np.cos(t), (R - 2.4 * k) * np.sin(t)) for t in np.arange(16) * np.pi / 8]
    top, seats = _seats(top, x, y, centres, STONE_SMALL, 0.9)
    Ri = R - 3.6 * k
    face = _face(head, s, 0.7 * 2 * Ri, 0.2 * 2 * Ri, 2 * Ri, depth * 0.85, log)
    top = top + _place(r.shape, face, -0.08 * Ri + s.face_shift, y0) * np.clip((Ri - 0.3 - r) / 0.4, 0, 1)
    top = np.where(body, top, 0)
    y_top = y[np.nonzero(body[:, body.shape[1] // 2])[0][0], 0]
    top, mask = _loop(top, body, x, y, y_top)
    return _finish(Piece("lumba", top, np.zeros_like(top), mask, seats, None), size, y0)


def _finish(p: Piece, size: float, y0: int) -> Piece:
    idx = ndimage.distance_transform_edt(~p.mask, return_distances=False, return_indices=True)
    p.top = np.where(p.mask, ndimage.gaussian_filter(p.top[tuple(idx)], 0.6), 0)
    p.bottom = np.where(p.mask, np.minimum(p.bottom, p.top - 0.6), 0)
    rows = np.nonzero(p.mask.any(1))[0]
    p.info = dict(size_mm=[round(size, 1), round((rows[-1] - rows[0]) * PX, 1)],
                  stones=len(p.seats), stone_mm=round(2 * p.seats[0][2], 1) if p.seats else 0,
                  thickness_mm=round(float((p.top - p.bottom)[p.mask].max()), 1), centre_row=y0)
    return p


def to_trimesh(p: Piece, log: list[str]):
    import manifold3d as mf
    import trimesh

    m = mesh_mod.heightmap_to_mesh(p.top, p.mask, pixel_mm=PX, bottom=p.bottom)
    t = trimesh.Trimesh(m.vertices, m.faces, process=False)
    if p.tunnel:
        row, c0, c1 = p.tunnel
        h = p.top.shape[0]
        yc = (h - 1 - row) * PX
        length = (c1 - c0) * PX + 2.0
        cyl = (mf.Manifold.cylinder(length, TUNNEL_R, TUNNEL_R, 48)
               .rotate([0, 90, 0]).translate([c0 * PX - 1.0, yc, -BOSS / 2]))
        t = royal._from_manifold(royal._to_manifold(t) - cyl)
        log.append(f"Thread tunnel: {2 * TUNNEL_R:.0f} mm across the back")
    t = royal.simplify(t, 450_000, 0.004)
    return royal.weld_short_edges(t, 0.01, log)


# ---------------------------------------------------------------- previews
def _stone_colours(s: RakhiSettings, kind: str):
    cols = STONES[s.stones]
    if kind == "lumba" and s.stones == "Red & green":
        cols = STONES["Pearl white"]
    return cols


def preview(p: Piece, s: RakhiSettings) -> Image.Image:
    """Front view with stones in their seats, on a rakhi thread (lumba: on its chain)."""
    h = p.top.copy()
    yy, xx = np.mgrid[0:h.shape[0], 0:h.shape[1]]
    gem_masks = []
    cols = _stone_colours(s, p.kind)
    for n, (row, col, rad) in enumerate(p.seats):
        rr = np.hypot(yy - row, xx - col) * PX
        g = rr < rad * 0.95
        h = np.where(g, h + _dome(rr, rad, rad * 0.8), h)
        gem_masks.append((g, cols[n % len(cols)]))
    mat = "gold"
    img = np.asarray(render.render_material(h, PX, p.mask, mat), float)
    lam = render.shade(h, PX)
    for g, col in gem_masks:
        c = np.array(col, float) * (0.35 + 0.75 * lam[..., None]) + (lam ** 30)[..., None] * 255
        img = np.where(g[..., None], np.clip(c, 0, 255), img)
    piece = Image.fromarray(np.dstack([img, p.mask * 255]).astype(np.uint8), "RGBA")
    W = piece.width + (500 if p.kind == "rakhi" else 120)
    H = piece.height + 120
    bg = Image.new("RGB", (W, H), (250, 244, 232))
    d = ImageDraw.Draw(bg)
    cy = 60 + p.info["centre_row"]
    if p.kind == "rakhi":
        for k in range(-9, 10, 4):
            d.line([(0, cy + k), (W, cy + k)], fill=(200, 30, 40), width=8)
        for xx_ in range(0, W, 26):
            d.line([(xx_, cy - 10), (xx_ + 16, cy + 10)], fill=(235, 190, 70), width=3)
    else:
        for i in range(6):
            d.ellipse([W // 2 - 10, 60 + 30 - 24 * (i + 1), W // 2 + 10, 60 + 30 - 24 * i],
                      outline=(205, 160, 70), width=5)
    bg.paste(piece, ((W - piece.width) // 2, 60), piece)
    return bg.resize((bg.width // 2, bg.height // 2), Image.LANCZOS)


def preview_back(p: Piece, s: RakhiSettings) -> Image.Image:
    h = -p.bottom[:, ::-1]
    return render.render_material(h, PX, p.mask[:, ::-1], "gold").resize(
        (p.mask.shape[1] // 3, p.mask.shape[0] // 3), Image.LANCZOS)


def making_notes(s: RakhiSettings, pieces: list[tuple[str, Piece]]) -> str:
    fdm = s.method.startswith("FDM")
    lines = ["Face Rakhi - making notes", "=" * 40, f"Design: {s.design} | {s.set_type} | {s.method}", ""]
    for label, p in pieces:
        lines.append(f"{label} ({p.kind}): {p.info['size_mm'][0]} x {p.info['size_mm'][1]} mm, "
                     f"{p.info['stones']} stones of {p.info['stone_mm']} mm (flat-back)")
    lines.append("")
    if fdm:
        lines += ["PRINT (FDM): silk gold PLA, 0.2 mm nozzle, 0.06-0.08 mm layers, 3 walls, 20% infill.",
                  "Orientation: face up, tree supports under the back ridge only (never on the face).",
                  "Or stand it on its edge with a brim - try both on the first test."]
    else:
        lines += ["PRINT (resin): fine-detail resin (e.g. LEDO 6060 / grey), 0.025-0.05 mm layers.",
                  "Supports on the back and loop only - never on the face.",
                  "PAINT: gold acrylic/metallic base, darker wash in the background for an antique look,",
                  "dry-brush bright gold on the face and rim. Clear coat."]
    lines += ["",
              "ASSEMBLE:",
              "1. Glue the stones into the seats (E6000 / Fevikwik gel; tweezers or a wax pencil).",
              "2. Rakhi: slide the dori (rakhi thread) through the tunnel on the back.",
              "3. Lumba: hang it from a lumba bangle / chain through the top loop.",
              "4. After the festival: slide the thread out, add a key ring to the top loop.",
              "",
              "Thread, stones, lumba bases and boxes: rakhi wholesale shops or online (Rs 2-10 each)."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- generate
def _pieces(sources, s: RakhiSettings, log) -> list[tuple[str, Piece]]:
    out = []
    (head1, label1) = sources[0]
    log.append(f"--- Rakhi ({label1}) ---")
    out.append((label1, build_rakhi(head1, s, log)))
    if s.set_type == SETS[1]:
        if len(sources) < 2:
            raise ValueError("The Bhaiya + Bhabhi set needs the second face (bhabhi) for the lumba.")
        head2, label2 = sources[1]
        log.append(f"--- Lumba ({label2}) ---")
        out.append((label2, build_lumba(head2, s, log)))
    return out


def quick_preview(sources, s: RakhiSettings) -> list[Path]:
    import tempfile

    out_dir = Path(tempfile.mkdtemp(prefix="rakhi_preview_"))
    paths = []
    for n, (label, p) in enumerate(_pieces(sources, s, [])):
        path = out_dir / f"preview_{n + 1}.png"
        preview(p, s).save(path)
        paths.append(path)
    return paths


def generate(sources: list[tuple[str, str]], s: RakhiSettings, customer: dict | None = None,
             api_key: str | None = None, progress=None) -> Result:
    """sources: [(brother's head, label)] (+ [(bhabhi's head, label)] for the set)."""
    from .providers import tripo

    customer = customer or {}
    log: list[str] = []
    t_all = time.time()
    check_text(s)
    order_id, folder = orders.new_order_dir("rakhi", customer.get("name", ""))
    heads = []
    for n, (src, label) in enumerate(sources):
        src = Path(src)
        tag = "".join(c for c in label.lower() if c.isalnum()) or f"p{n + 1}"
        if src.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            head = tripo.photo_to_head_model(src, api_key or "", folder, log, progress=progress)
        else:
            head = folder / f"head_{tag}{src.suffix.lower()}"
            head.write_bytes(src.read_bytes())
        heads.append((head, label))
    if progress:
        progress(0.3, desc="Building the rakhi...")
    pieces = _pieces(heads, s, log)
    files, previews, vols, glb = [], [], [], None
    for label, p in pieces:
        tag = "".join(c for c in label.lower() if c.isalnum()) + f"_{p.kind}"
        t = to_trimesh(p, log)
        if not t.is_watertight:
            log.append("WARNING: mesh is not watertight - check before printing")
        stl = folder / f"{order_id}_{tag}.stl"
        t.export(stl)
        vols.append(float(abs(t.volume)))
        img = folder / f"{order_id}_{tag}.png"
        preview(p, s).save(img)
        back = folder / f"{order_id}_{tag}_back.png"
        preview_back(p, s).save(back)
        files += [stl, img, back]
        previews.append(img)
        log.append(f"{label} {p.kind}: {p.info['size_mm'][0]} x {p.info['size_mm'][1]} mm, "
                   f"{p.info['stones']} stones, volume {vols[-1] / 1000:.1f} cm3")
        if glb is None:
            st = 6
            small = mesh_mod.heightmap_to_mesh(p.top[::st, ::st], p.mask[::st, ::st], PX * st,
                                               bottom=p.bottom[::st, ::st])
            glb = mesh_mod.write_glb_preview(small, folder / "preview.glb", color=(225, 185, 95))
    notes = folder / f"{order_id}_making_notes.txt"
    notes.write_text(making_notes(s, pieces), encoding="utf-8")
    files.append(notes)
    q = costing.quote_rakhi(vols, [p.info["stones"] for _, p in pieces], s.method,
                            [p.kind for _, p in pieces], s.packaging)
    gst = load_pricing()["gst_percent"]
    orders.save_order(folder, {"order_id": order_id, "product": "face rakhi", "customer": customer,
                               "quantity": 1, "settings": asdict(s),
                               "labels": [lbl for lbl, _ in pieces],
                               "quote": {"price_per_piece": q.price_per_piece, "total_price": q.total_price,
                                         "lines": q.lines},
                               "status": "proof sent", "log": log})
    log.append(f"Total time {time.time() - t_all:.0f}s")
    return Result(order_id, folder, files, previews, glb, q.as_markdown(gst), log)
