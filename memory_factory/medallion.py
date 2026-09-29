"""Wedding return-gift medallion: couple photo -> printable coin STL.

Pipeline
    photo -> background removal -> depth (AI or heuristic)
          -> bas-relief compression -> coin layout (portrait, rim, names, date)
          -> watertight mesh -> STL + 3D preview + WhatsApp proof card + quote
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from . import costing, depth as depth_mod, imaging, mesh as mesh_mod, orders, relief, render


LAYOUTS = ["coin", "classic"]


@dataclass
class MedallionSettings:
    names: str = "Ramesh & Meera"
    date: str = "22.04.2026"
    extra: str = ""               # optional message (top arc in "coin" layout)
    layout: str = "coin"          # coin: big portrait + curved rim text | classic
    diameter_mm: float = 50.0
    base_mm: float = 2.5          # flat coin thickness under the relief
    relief_mm: float = 1.5        # tallest point of the faces above the field
    text_mm: float = 0.7          # raised height of letters
    rim_width_mm: float = 2.0
    keychain_hole: bool = False
    pixel_mm: float = 0.15        # mesh resolution (0.1 = finer, bigger file)
    remove_background: bool = True
    depth_engine: str = "auto"    # auto | ai | fast
    compression: float = 0.45
    detail: float = 1.0
    material: str = "antique brass"
    font_path: str | None = None


@dataclass
class Result:
    order_id: str
    folder: Path
    stl: Path
    glb: Path | None
    render: Path
    proof: Path
    depth_preview: Path
    quote_md: str
    log: list[str]
    volume_mm3: float
    triangles: int
    warnings: list[str]


def _smooth_disk(size: int, radius: float, soft: float = 1.5) -> np.ndarray:
    c = (size - 1) / 2
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(xx - c, yy - c)
    return np.clip((radius - r) / soft + 0.5, 0, 1)


def _resize(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(arr.astype(np.float32)).resize((w, h), Image.BICUBIC),
                      dtype=np.float64)


def _prepare_portrait(photo: Image.Image, s: MedallionSettings, log: list[str],
                      warnings: list[str]):
    """Cut out the people, find faces, crop, estimate depth."""
    t0 = time.time()
    img = imaging.limit_size(photo, 1024)
    mask, eng = imaging.subject_mask(img, use_ai=s.remove_background)
    log.append(f"Background: {eng} ({time.time() - t0:.1f}s)")
    if s.remove_background and "AI" not in eng:
        warnings.append("AI background removal is not installed/working - the background may "
                        "show on the coin. Run: pip install -r requirements.txt")

    faces = imaging.detect_faces(img, mask)
    log.append(f"Faces found: {len(faces)}")
    if not faces:
        warnings.append("No faces detected - check the photo (faces clear, looking at the camera).")

    if s.layout == "coin":
        x0, y0, x1, y1 = imaging.portrait_crop_box(img.size, mask, faces, aspect=1.1)
        img_c, mask_c = img.crop((x0, y0, x1, y1)), mask[y0:y1, x0:x1]
        log.append("Auto-crop: head and shoulders")
    else:
        ys, xs = np.nonzero(mask > 0.5)
        x0, y0 = (xs.min(), ys.min()) if len(xs) else (0, 0)
        img_c, mask_c = imaging.crop_to_mask(img, mask)
        # crop_to_mask pads the box; recover its real origin from the size change.
        x0 = max(0, x0 - int((xs.max() - xs.min()) * 0.06)) if len(xs) else 0
        y0 = max(0, y0 - int((ys.max() - ys.min()) * 0.06)) if len(ys) else 0
    faces_c = [(x - x0, y - y0, w, h) for x, y, w, h in faces]

    t0 = time.time()
    engine = {"fast": "heuristic"}.get(s.depth_engine, s.depth_engine)
    d, eng = depth_mod.estimate_depth(img_c, mask_c, engine=engine, faces=faces_c)
    log.append(f"Depth: {eng} ({time.time() - t0:.1f}s)")
    return img_c, mask_c, faces_c, d


def _portrait_relief(img_c, mask_c, faces_c, d, pw, ph, s: MedallionSettings, log):
    ih, iw = mask_c.shape
    sx, sy = pw / iw, ph / ih
    d_s, m_s = _resize(d, pw, ph), np.clip(_resize(mask_c, pw, ph), 0, 1)
    g_s = _resize(imaging.to_gray(img_c), pw, ph)
    if faces_c:
        faces_s = [(x * sx, y * sy, w * sx, h * sy) for x, y, w, h in faces_c]
        fw = imaging.face_weight_map((ph, pw), faces_s)
        detail = s.detail * (0.3 + 0.7 * fw)   # sharp faces, calm clothes
    else:
        detail = s.detail
    t0 = time.time()
    rel = relief.bas_relief(d_s, m_s, g_s, compression=s.compression, detail=detail)
    log.append(f"Relief compression ({time.time() - t0:.1f}s)")
    # Where the people are cut by the photo edge, fade out instead of a hard wall.
    ramp_px = max(2.0, 0.07 * pw)
    xs = np.arange(pw)
    fx = np.clip(np.minimum(xs, pw - 1 - xs) / ramp_px, 0, 1)
    fy = np.clip((ph - 1 - np.arange(ph)) / ramp_px, 0, 1)
    fade = np.minimum(fx[None, :], fy[:, None])
    return rel * fade * fade * (3 - 2 * fade)


def _arc_text(n, c, text, r_mid, text_h, where, font_path, max_span_deg=150.0):
    """Text bent along a circle. Bottom: reads left->right, letter tops inward.
    Top: reads left->right, letter tops outward (like a coin)."""
    strip = imaging.text_strip(text, int(text_h), font_path)
    if strip.shape[1] <= 2:
        return np.zeros((n, n))
    span = strip.shape[1] / r_mid
    max_span = np.radians(max_span_deg)
    if span > max_span:  # too long - shrink the letters
        k = max_span / span
        strip = _resize(strip, max(2, int(strip.shape[1] * k)), max(2, int(strip.shape[0] * k)))
        span = strip.shape[1] / r_mid
    sh, sw = strip.shape
    yy, xx = np.mgrid[0:n, 0:n]
    dx, dy = xx - c, yy - c
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)  # image y points down: bottom = +pi/2
    if where == "bottom":
        u = (np.pi / 2 + span / 2 - theta) / span * sw
        v = r - (r_mid - sh / 2)
    else:
        u = (theta - (-np.pi / 2 - span / 2)) / span * sw
        v = (r_mid + sh / 2) - r
    inside = (u >= 0) & (u <= sw - 1) & (v >= 0) & (v <= sh - 1)
    out = np.zeros((n, n))
    out[inside] = ndimage.map_coordinates(strip, [v[inside], u[inside]], order=1)
    return out


def build_heightmap(photo: Image.Image, s: MedallionSettings, log: list[str],
                    warnings: list[str] | None = None):
    warnings = [] if warnings is None else warnings
    px = s.pixel_mm
    n = int(round(s.diameter_mm / px)) | 1
    R = (n - 1) / 2
    c = R
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - c, yy - c)

    coin = rr <= R
    rim_in = R - s.rim_width_mm / px
    inner = _smooth_disk(n, rim_in - 1.5, soft=2.0)

    img_c, mask_c, faces_c, d = _prepare_portrait(photo, s, log, warnings)
    ih, iw = mask_c.shape
    portrait = np.zeros((n, n))
    text = np.zeros((n, n))
    extra_rings = np.zeros((n, n))

    if s.layout == "coin":
        # Rim text band, a fine bead ring, and a big portrait disc inside it.
        band = max(3.0 / px, rim_in * 0.2)
        rb = rim_in - band
        rp = rb - 0.5 / px
        pw = int(2 * rp)
        ph = int(pw * ih / iw)
        if ph > 1.9 * rp:
            ph = int(1.9 * rp)
            pw = int(ph * iw / ih)
        rel = _portrait_relief(img_c, mask_c, faces_c, d, pw, ph, s, log)
        # Shoulders fade softly into the field over the lower 30%.
        t = np.clip((np.arange(ph) - ph * 0.7) / (ph * 0.3), 0, 1)
        rel *= (1 - t * t * (3 - 2 * t))[:, None]
        x0 = int(c - pw / 2)
        y0 = int(c - rp * 0.97)
        y1 = min(n, y0 + ph)
        portrait[y0:y1, x0:x0 + pw] = rel[: y1 - y0]
        portrait *= _smooth_disk(n, rp, soft=3.0)

        text_h = band * 0.74
        r_mid = rb + band * 0.52
        bottom = s.names
        top = " \u2022 ".join(t for t in (s.extra, s.date) if t)
        if s.keychain_hole:  # top arc is taken by the hole
            bottom = " \u2022 ".join(t for t in (s.names, s.date) if t)
            top = ""
        text = np.maximum(text, _arc_text(n, c, bottom, r_mid, text_h, "bottom", s.font_path))
        if top:
            text = np.maximum(text, _arc_text(n, c, top, r_mid, text_h * 0.8, "top",
                                              s.font_path, max_span_deg=120))
        text = ndimage.gaussian_filter(text, 0.6)
        bead = np.clip(1 - np.abs(rr - rb) / (0.3 / px), 0, 1)
        extra_rings = bead * bead * (3 - 2 * bead) * 0.8
        log.append(f"Layout: coin - portrait {2 * rp * px:.0f} mm wide, curved rim text")
    else:
        box_top = c - rim_in * (0.66 if s.keychain_hole else 0.86)
        box_bot = c + rim_in * 0.34
        scale = min(rim_in * 1.55 / iw, (box_bot - box_top) / ih)
        pw, ph = max(2, int(iw * scale)), max(2, int(ih * scale))
        rel = _portrait_relief(img_c, mask_c, faces_c, d, pw, ph, s, log)
        x0, y0 = int(c - pw / 2), int(box_bot - ph)
        portrait[y0:y0 + ph, x0:x0 + pw] = rel
        portrait *= inner
        rows = [(s.names, 0.40, 0.60, 1.0), (s.date, 0.62, 0.74, 0.8), (s.extra, 0.76, 0.86, 0.7)]
        for line, top_f, bot_f, width_f in rows:
            if not line:
                continue
            ty0, ty1 = int(c + rim_in * top_f), int(c + rim_in * bot_f)
            mid = (top_f + bot_f) / 2
            tw = int(2 * np.sqrt(max(0.0, 1 - mid**2)) * rim_in * 0.92 * width_f)
            tm = imaging.text_mask(line, tw, ty1 - ty0, font_path=s.font_path)
            tx0 = int(c - tw / 2)
            text[ty0:ty1, tx0:tx0 + tw] = np.maximum(text[ty0:ty1, tx0:tx0 + tw], tm)
        text = ndimage.gaussian_filter(text, 0.6) * inner
        log.append("Layout: classic")

    # ---- rim --------------------------------------------------------------
    rim_h = max(s.relief_mm * 0.75, s.text_mm + 0.2)
    rim = np.clip((rr - rim_in) / 3.0, 0, 1)
    rim = rim * rim * (3 - 2 * rim)
    edge = np.clip((R - rr) / 2.0, 0, 1)  # small bevel on the outside edge
    rim = rim * (0.6 + 0.4 * edge)

    height = (s.base_mm + portrait * s.relief_mm + np.maximum(text, extra_rings) * s.text_mm
              + rim * rim_h)
    height = np.maximum(height, s.base_mm)

    if s.keychain_hole:
        hole_r = 1.6 / px
        hy = c - rim_in + hole_r + 1.2 / px
        coin &= np.hypot(xx - c, yy - hy) > hole_r
        boss = np.clip((hole_r + 1.5 / px - np.hypot(xx - c, yy - hy)) / 2, 0, 1)
        height = np.maximum(height, s.base_mm + boss * rim_h)
        log.append("Keychain hole added (3.2 mm).")

    return height, coin


def generate(photo, settings: MedallionSettings, customer: dict | None = None,
             quantity: int = 100, finish: str = "antique brass paint",
             packaging: str = "velvet pouch") -> Result:
    log: list[str] = []
    t_all = time.time()
    photo = imaging.load_image(photo)
    customer = customer or {}
    order_id, folder = orders.new_order_dir("medallion", customer.get("name", ""))

    warnings: list[str] = []
    # Design at a fine resolution (sharp proof image), mesh at settings.pixel_mm.
    px = settings.pixel_mm
    k = max(1, int(round(px / 0.075)))
    fine = replace(settings, pixel_mm=px / k)
    height_fine, coin_fine = build_heightmap(photo, fine, log, warnings)
    if k > 1:
        n = int(round(settings.diameter_mm / px)) | 1

        def down(arr):
            return np.asarray(Image.fromarray(arr.astype(np.float32)).resize((n, n), Image.BOX),
                              dtype=np.float64)

        height = down(height_fine)
        coin = down(coin_fine) > 0.5
    else:
        height, coin = height_fine, coin_fine

    t0 = time.time()
    solid = mesh_mod.heightmap_to_mesh(height, coin, px, bottom=0.0)
    stl = mesh_mod.write_stl(solid, folder / f"{order_id}_medallion.stl")
    log.append(f"Mesh: {solid.triangle_count:,} triangles, watertight="
               f"{mesh_mod.is_watertight(solid)} ({time.time() - t0:.1f}s)")

    step = max(1, int(round(0.4 / px)))
    small = mesh_mod.heightmap_to_mesh(height[::step, ::step], coin[::step, ::step], px * step)
    render_img = render.render_material(height_fine, px / k, coin_fine, settings.material)
    ext = (height_fine.shape[1] - 1) * px / k
    glb = mesh_mod.write_glb_preview(small, folder / "preview.glb", texture=render_img,
                                     extent_mm=(ext, ext), upright=True)

    render_path = folder / "render.png"
    render_img.save(render_path)
    depth_prev = folder / "relief_heightmap.png"
    hm = (height - height.min()) / (np.ptp(height) + 1e-9)
    Image.fromarray((hm * 255).astype(np.uint8)).save(depth_prev)
    photo.save(folder / "customer_photo.jpg", quality=92)

    vol = solid.volume_mm3()
    q = costing.quote_medallion(vol, quantity, finish, packaging)
    gst = costing.load_pricing()["gst_percent"]
    proof = render.proof_sheet(
        render_img,
        f"Return gift medallion - {settings.names}",
        [
            f"Order: {order_id}",
            f"Size: {settings.diameter_mm:.0f} mm, finish: {finish}",
            f"Quantity: {quantity}   Packaging: {packaging}",
            f"Names: {settings.names}",
            f"Date / message: {settings.date}" + (f"  |  {settings.extra}" if settings.extra else ""),
            f"Price: Rs {q.price_per_piece:,.0f} x {q.quantity} + Rs {q.setup_fee:,.0f} design"
            f" = Rs {q.total_price:,.0f}",
        ],
    )
    proof_path = folder / "proof_for_customer.jpg"
    proof.save(proof_path, quality=90)

    orders.save_order(folder, {
        "order_id": order_id,
        "product": "wedding medallion",
        "customer": customer,
        "quantity": quantity,
        "finish": finish,
        "packaging": packaging,
        "settings": {k: v for k, v in asdict(settings).items() if k != "font_path"},
        "volume_ml": round(vol / 1000, 2),
        "quote": {"cost_per_piece": round(q.cost_per_piece, 2),
                  "price_per_piece": q.price_per_piece, "setup_fee": q.setup_fee,
                  "total_price": q.total_price},
        "status": "proof sent",
        "log": log,
        "warnings": warnings,
    })
    log.append(f"Resin per coin ≈ {vol / 1000:.1f} ml. Total time {time.time() - t_all:.1f}s.")
    return Result(order_id, folder, stl, glb, render_path, proof_path, depth_prev,
                  q.as_markdown(gst), log, vol, solid.triangle_count, warnings)
