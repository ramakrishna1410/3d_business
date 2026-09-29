"""Wedding return-gift medallion: couple photo -> printable coin STL.

Pipeline
    photo -> background removal -> depth (AI or heuristic)
          -> bas-relief compression -> coin layout (portrait, rim, names, date)
          -> watertight mesh -> STL + 3D preview + WhatsApp proof card + quote
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from . import costing, depth as depth_mod, imaging, mesh as mesh_mod, orders, relief, render


@dataclass
class MedallionSettings:
    names: str = "Ramesh & Meera"
    date: str = "22.04.2026"
    extra: str = "With Love & Thanks"
    diameter_mm: float = 50.0
    base_mm: float = 2.5          # flat coin thickness under the relief
    relief_mm: float = 1.4        # tallest point of the faces above the field
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


def _smooth_disk(size: int, radius: float, soft: float = 1.5) -> np.ndarray:
    c = (size - 1) / 2
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(xx - c, yy - c)
    return np.clip((radius - r) / soft + 0.5, 0, 1)


def build_heightmap(photo: Image.Image, s: MedallionSettings, log: list[str]):
    px = s.pixel_mm
    n = int(round(s.diameter_mm / px)) | 1
    R = (n - 1) / 2
    c = R
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - c, yy - c)

    coin = rr <= R
    rim_in = R - s.rim_width_mm / px
    inner = _smooth_disk(n, rim_in - 1.5, soft=2.0)

    # ---- portrait -----------------------------------------------------
    t0 = time.time()
    img = imaging.limit_size(photo, 1024)
    mask, eng = imaging.subject_mask(img, use_ai=s.remove_background)
    log.append(f"Background: {eng} ({time.time() - t0:.1f}s)")
    img_c, mask_c = imaging.crop_to_mask(img, mask)

    t0 = time.time()
    engine = {"fast": "heuristic"}.get(s.depth_engine, s.depth_engine)
    d, eng = depth_mod.estimate_depth(img_c, mask_c, engine=engine)
    log.append(f"Depth: {eng} ({time.time() - t0:.1f}s)")

    # Portrait box: upper part of the coin.
    box_top = c - rim_in * (0.66 if s.keychain_hole else 0.86)
    box_bot = c + rim_in * 0.34
    box_h = box_bot - box_top
    box_w = rim_in * 1.55
    ih, iw = mask_c.shape
    scale = min(box_w / iw, box_h / ih)
    pw, ph = max(2, int(iw * scale)), max(2, int(ih * scale))

    def fit(arr):
        return np.asarray(Image.fromarray(arr.astype(np.float32)).resize((pw, ph), Image.BICUBIC),
                          dtype=np.float64)

    d_s, m_s = fit(d), np.clip(fit(mask_c), 0, 1)
    g_s = fit(imaging.to_gray(img_c))

    t0 = time.time()
    rel = relief.bas_relief(d_s, m_s, g_s, compression=s.compression, detail=s.detail)
    log.append(f"Relief compression ({time.time() - t0:.1f}s)")

    portrait = np.zeros((n, n))
    x0 = int(c - pw / 2)
    y0 = int(box_bot - ph)  # sit the people on the text band
    portrait[y0:y0 + ph, x0:x0 + pw] = rel
    portrait *= inner

    # ---- text -----------------------------------------------------------
    text = np.zeros((n, n))
    rows = [(s.names, 0.40, 0.60, 1.0), (s.date, 0.62, 0.74, 0.8), (s.extra, 0.76, 0.86, 0.7)]
    for line, top_f, bot_f, width_f in rows:
        if not line:
            continue
        ty0, ty1 = int(c + rim_in * top_f), int(c + rim_in * bot_f)
        mid = (top_f + bot_f) / 2
        chord = 2 * np.sqrt(max(0.0, 1 - mid**2)) * rim_in * 0.92 * width_f
        tw = int(chord)
        tm = imaging.text_mask(line, tw, ty1 - ty0, font_path=s.font_path)
        tx0 = int(c - tw / 2)
        text[ty0:ty1, tx0:tx0 + tw] = np.maximum(text[ty0:ty1, tx0:tx0 + tw], tm)
    text = ndimage.gaussian_filter(text, 0.6) * inner

    # ---- rim --------------------------------------------------------------
    rim_h = max(s.relief_mm * 0.75, s.text_mm + 0.2)
    rim = np.clip((rr - rim_in) / 3.0, 0, 1)
    rim = rim * rim * (3 - 2 * rim)
    edge = np.clip((R - rr) / 2.0, 0, 1)  # small bevel on the outside edge
    rim = rim * (0.6 + 0.4 * edge)

    height = s.base_mm + portrait * s.relief_mm + text * s.text_mm + rim * rim_h
    height = np.maximum(height, s.base_mm)

    if s.keychain_hole:
        hole_r = 1.6 / px
        hy = c - rim_in + hole_r + 2.2 / px
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

    height, coin = build_heightmap(photo, settings, log)
    px = settings.pixel_mm

    t0 = time.time()
    solid = mesh_mod.heightmap_to_mesh(height, coin, px, bottom=0.0)
    stl = mesh_mod.write_stl(solid, folder / f"{order_id}_medallion.stl")
    log.append(f"Mesh: {solid.triangle_count:,} triangles, watertight="
               f"{mesh_mod.is_watertight(solid)} ({time.time() - t0:.1f}s)")

    step = max(1, int(round(0.4 / px)))
    small = mesh_mod.heightmap_to_mesh(height[::step, ::step], coin[::step, ::step], px * step)
    glb = mesh_mod.write_glb_preview(small, folder / "preview.glb")

    render_img = render.render_material(height, px, coin, settings.material)
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
            f"Date / message: {settings.date}  |  {settings.extra}",
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
    })
    log.append(f"Resin per coin ≈ {vol / 1000:.1f} ml. Total time {time.time() - t_all:.1f}s.")
    return Result(order_id, folder, stl, glb, render_path, proof_path, depth_prev,
                  q.as_markdown(gst), log, vol, solid.triangle_count)
