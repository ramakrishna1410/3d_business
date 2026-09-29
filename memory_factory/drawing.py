"""Kids' drawing -> 3D sculpture.

Three modes:

* "relief plaque"   : the drawing is puffed up on a rectangular plaque with
                      the child's name. Fully offline, strongest, cheapest.
* "standing figure" : the drawing's outline is inflated on BOTH sides into
                      a free-standing figure, glued into a name base.
                      Fully offline ("Crayon Creatures" style).
* "ai full 3d"      : the drawing is sent to Meshy image-to-3D (needs an API
                      key, paid credits). The returned model is scaled,
                      put on a name base and exported as STL.
* "line-art relief" : for outline sketches (e.g. a Ganesha on a peepal leaf).
                      The plaque takes the drawing's own outline shape; areas
                      touching the outline form the background layer, inner
                      areas form a raised figure layer. See lineart.py.
* "import 3d model" : you generated the model yourself on Tripo / Meshy's
                      website and downloaded it (GLB/OBJ/STL). Same clean-up,
                      scaling and name base as above - no API key needed.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from . import costing, imaging, mesh as mesh_mod, orders, relief, render

MODES = ["relief plaque", "standing figure", "line-art relief", "ai full 3d", "import 3d model"]


@dataclass
class DrawingSettings:
    child_name: str = "Aarav"
    age_line: str = "Age 6 - 2026"
    mode: str = "relief plaque"
    size_mm: float = 100.0         # longest side of the drawing
    puff_mm: float = 6.0           # how "fat" the inflated shape is
    line_depth_mm: float = 0.5     # grooves along the child's pencil lines
    plaque_mm: float = 3.0         # plaque thickness
    roundness: float = 1.0
    pixel_mm: float = 0.25
    font_path: str | None = None


@dataclass
class Result:
    order_id: str
    folder: Path
    stl: Path
    glb: Path | None
    render: Path
    painting_guide: Path
    quote_md: str
    log: list[str]
    volume_mm3: float
    height_mm: float


def _line_grooves(img: Image.Image, mask: np.ndarray) -> np.ndarray:
    """Dark pencil/crayon outlines inside the shape (0..1)."""
    gray = imaging.to_gray(img)
    local = ndimage.gaussian_filter(gray, 6)
    lines = np.clip((local - gray - 0.08) * 5, 0, 1)
    lines = ndimage.binary_opening(lines > 0.3, iterations=1).astype(float)
    inner = ndimage.binary_erosion(mask, iterations=4)
    return ndimage.gaussian_filter(lines * inner, 0.8)


def _prepare(drawing: Image.Image, s: DrawingSettings, log: list[str]):
    img = imaging.limit_size(drawing, 1400)
    mask = imaging.drawing_mask(img)
    if mask.mean() < 0.005:
        raise ValueError("Could not find the drawing on the paper. Use a photo with the "
                         "drawing on plain white paper, good light, no shadows.")
    ys, xs = np.nonzero(mask)
    pad = 8
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad)
    img, mask = img.crop((x0, y0, x1, y1)), mask[y0:y1, x0:x1]

    # Resample so one pixel = pixel_mm and the longest side = size_mm.
    longest = max(mask.shape)
    scale = (s.size_mm / s.pixel_mm) / longest
    w, h = max(2, int(mask.shape[1] * scale)), max(2, int(mask.shape[0] * scale))
    img = img.resize((w, h), Image.LANCZOS)
    mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.BILINEAR)) > 127
    mask = ndimage.binary_fill_holes(mask)
    log.append(f"Drawing found: {w * s.pixel_mm:.0f} x {h * s.pixel_mm:.0f} mm, "
               f"{mask.mean() * 100:.0f}% of the area")
    return img, mask


def _name_band(width_px: int, s: DrawingSettings, band_mm: float = 18.0) -> np.ndarray:
    band_h = int(band_mm / s.pixel_mm)
    a = imaging.text_mask(s.child_name, width_px, int(band_h * 0.6), s.font_path)
    b = imaging.text_mask(s.age_line, width_px, band_h - int(band_h * 0.6), s.font_path)
    return np.vstack([a, b])


def _pad(a: np.ndarray, p: int, value=0):
    return np.pad(a, p, mode="constant", constant_values=value)


def build_relief_plaque(img, mask, s: DrawingSettings):
    px = s.pixel_mm
    dome = relief.inflate(mask, s.roundness)
    grooves = _line_grooves(img, mask)
    shape_h = dome * s.puff_mm - grooves * s.line_depth_mm
    shape_h = np.where(mask, np.maximum(shape_h, 0.6), 0)

    margin = int(8 / px)
    shape_h = _pad(shape_h, margin)
    colour = np.asarray(img.convert("RGB"))
    colour = np.pad(colour, ((margin, margin), (margin, margin), (0, 0)), constant_values=255)
    body_mask = _pad(mask, margin, False)

    name = _name_band(shape_h.shape[1] - 2 * margin, s)
    name = np.pad(name, ((0, margin // 2), (margin, margin)))
    text_h = ndimage.gaussian_filter(name, 0.7) * 1.2
    height = np.vstack([shape_h, text_h]) + s.plaque_mm
    plate = np.ones(height.shape, dtype=bool)

    # Rounded corners.
    r = margin
    hh, ww = plate.shape
    yy, xx = np.mgrid[0:hh, 0:ww]
    for cy, cx in [(r, r), (r, ww - r - 1), (hh - r - 1, r), (hh - r - 1, ww - r - 1)]:
        corner = ((yy < r) if cy == r else (yy > hh - r - 1)) & ((xx < r) if cx == r else (xx > ww - r - 1))
        plate &= ~(corner & (np.hypot(yy - cy, xx - cx) > r))

    colour_full = np.vstack([colour, np.full((name.shape[0], colour.shape[1], 3), 235, np.uint8)])
    solid = mesh_mod.heightmap_to_mesh(height, plate, px, bottom=0.0)
    vis_mask = np.vstack([body_mask, name > 0.3])
    return solid, height, plate, Image.fromarray(colour_full), vis_mask


def build_standing_figure(img, mask, s: DrawingSettings):
    px = s.pixel_mm
    dome = relief.inflate(mask, s.roundness)
    grooves = _line_grooves(img, mask)
    half = dome * (max(s.puff_mm, 8.0) / 2) - grooves * s.line_depth_mm * 0.6
    half = np.where(mask, np.maximum(half, 1.2), 0)  # >= 2.4 mm total everywhere
    m = _pad(mask, 2, False)
    half = _pad(half, 2)
    figure = mesh_mod.heightmap_to_mesh(half, m, px, bottom=-half)

    # Stand it up: drawing plane (x, y) -> (x, z); thickness goes along y.
    fig = figure.rotated_x(90)
    vmin, vmax = fig.bounds()
    width = vmax[0] - vmin[0]
    depth = max(22.0, s.puff_mm * 3)

    # Name plate: thin slab with raised letters, glued to the front of the base.
    name = _name_band(int((width + 8) / px), s, band_mm=9.0)
    name_h = ndimage.gaussian_filter(name, 0.7) * 1.0 + 1.0
    nm = np.ones(name.shape, dtype=bool)
    slab = mesh_mod.heightmap_to_mesh(_pad(name_h, 1, 1.0), _pad(nm, 1, True), px, bottom=0.0)
    slab = slab.rotated_x(90)  # letters now face the viewer (-y)
    sb0, sb1 = slab.bounds()
    slab_h = sb1[2] - sb0[2]

    base_h, sink = max(8.0, slab_h + 1.0), 2.0
    base = mesh_mod.box((width + 12, depth, base_h))
    slab = slab.translated([-sb0[0] + 2, -sb1[1] + 0.8, -sb0[2] + (base_h - slab_h) / 2])
    fig = fig.translated([-vmin[0] + 6, -(vmin[1] + vmax[1]) / 2 + depth / 2,
                          -vmin[2] + base_h - sink])

    solid = mesh_mod.combine([base, fig, slab])
    return solid, half, m


def generate(drawing, settings: DrawingSettings, customer: dict | None = None,
             paint: str = "hand painted", packaging: str = "display box",
             meshy_api_key: str | None = None, model_file=None, progress=None) -> Result:
    log: list[str] = []
    t_all = time.time()
    drawing = imaging.load_image(drawing)
    customer = customer or {}
    order_id, folder = orders.new_order_dir("drawing", customer.get("name", settings.child_name))
    drawing.save(folder / "original_drawing.jpg", quality=92)
    px = settings.pixel_mm

    preview_solid = None
    preview_tex = preview_extent = None  # shaded render painted onto flat products
    if settings.mode == "line-art relief":
        from . import lineart

        la = lineart.LineArtSettings(size_mm=settings.size_mm, pixel_mm=min(px, 0.2),
                                     groove_mm=settings.line_depth_mm)
        height, mask, img = lineart.build(drawing, la, log)
        solid = mesh_mod.heightmap_to_mesh(height, mask, la.pixel_mm)
        preview_solid = mesh_mod.heightmap_to_mesh(height[::2, ::2], mask[::2, ::2],
                                                   la.pixel_mm * 2)
        render_img = render.render_material(height, la.pixel_mm, mask, "antique brass")
        preview_tex = render_img
        preview_extent = ((height.shape[1] - 1) * la.pixel_mm, (height.shape[0] - 1) * la.pixel_mm)
    else:
        img, mask = _prepare(drawing, settings, log)

    if settings.mode == "line-art relief":
        pass
    elif settings.mode in ("ai full 3d", "import 3d model"):
        from .providers import meshy

        def band(width_mm):
            return _name_band(int(width_mm / px), settings, band_mm=9.0)

        if settings.mode == "ai full 3d":
            solid = meshy.image_to_printable_mesh(
                img, api_key=meshy_api_key, target_height_mm=settings.size_mm,
                name_band=band, pixel_mm=px, out_dir=folder, log=log, progress=progress)
        else:
            if not model_file:
                raise ValueError("Upload the GLB/OBJ/STL you downloaded from Tripo or Meshy.")
            solid = meshy.model_file_to_printable_mesh(
                model_file, settings.size_mm, band, px, log)
        render_img = render.render_coloured(relief.inflate(mask) * settings.puff_mm, px, img, mask)
    elif settings.mode == "standing figure":
        solid, half, m = build_standing_figure(img, mask, settings)
        colour = Image.fromarray(np.pad(np.asarray(img), ((2, 2), (2, 2), (0, 0)),
                                        constant_values=255))
        render_img = render.render_coloured(half, px, colour, m)
    else:
        solid, height, plate, colour, vis = build_relief_plaque(img, mask, settings)
        render_img = render.render_coloured(height, px, colour, plate)
        preview_solid = mesh_mod.heightmap_to_mesh(height[::2, ::2], plate[::2, ::2], px * 2)
        preview_tex = render_img
        preview_extent = ((height.shape[1] - 1) * px, (height.shape[0] - 1) * px)

    stl = mesh_mod.write_stl(solid, folder / f"{order_id}_{settings.mode.replace(' ', '_')}.stl")
    log.append(f"Mesh: {solid.triangle_count:,} triangles, watertight shells="
               f"{mesh_mod.is_watertight(solid)}")
    lo, hi = solid.bounds()
    size = hi - lo
    log.append(f"Final size: {size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} mm")

    glb = mesh_mod.write_glb_preview(preview_solid or solid, folder / "preview.glb",
                                     color=(235, 200, 120), texture=preview_tex,
                                     extent_mm=preview_extent, upright=preview_tex is not None)
    render_path = folder / "render.png"
    render_img.save(render_path)
    guide = render.painting_guide(img, mask)
    guide_path = folder / "painting_guide.png"
    guide.save(guide_path)

    vol = solid.volume_mm3()
    flat = settings.mode in ("relief plaque", "line-art relief")
    print_height = min(size) if flat else size[2]
    q = costing.quote_drawing(vol, print_height, paint, packaging)
    gst = costing.load_pricing()["gst_percent"]

    orders.save_order(folder, {
        "order_id": order_id,
        "product": f"kids drawing - {settings.mode}",
        "customer": customer,
        "quantity": 1,
        "paint": paint,
        "packaging": packaging,
        "settings": {k: v for k, v in asdict(settings).items() if k != "font_path"},
        "size_mm": [round(float(v), 1) for v in size],
        "volume_ml": round(vol / 1000, 2),
        "quote": {"cost_per_piece": round(q.cost_per_piece, 2),
                  "price_per_piece": q.price_per_piece, "total_price": q.total_price},
        "status": "proof sent",
        "log": log,
    })
    log.append(f"Resin ≈ {vol / 1000:.1f} ml. Total time {time.time() - t_all:.1f}s.")
    return Result(order_id, folder, stl, glb, render_path, guide_path, q.as_markdown(gst),
                  log, vol, float(size[2]))
