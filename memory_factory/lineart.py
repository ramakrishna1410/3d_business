"""Line-art (pencil / pen sketch) -> layered relief plaque.

Made for outline drawings such as a Ganesha on a peepal leaf, rangoli or
kolam designs, or a child's outline drawing. The drawing is split into the
areas enclosed by the pencil lines, and:

* the outer outline becomes the plaque's shape (leaf, heart, temple...)
  with a raised rim
* areas that touch the outer outline form the BACKGROUND layer (the leaf)
* areas away from it form the FIGURE layer (the Ganesha), raised higher
* every area is gently rounded ("pillowed"): bigger areas puff up more
* the pencil lines are pressed in as grooves, darker lines = deeper,
  so shading and hatching show up as texture

Small gaps in the pencil lines are bridged automatically so a figure does
not "leak" into the background.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from . import imaging


@dataclass
class LineArtSettings:
    size_mm: float = 150.0        # longest side of the plaque
    pixel_mm: float = 0.2
    base_mm: float = 3.0          # flat backing thickness
    background_mm: float = 0.8    # background (leaf) areas above the backing
    figure_mm: float = 2.5        # figure layer above the backing
    puff_mm: float = 3.0          # extra rounding of the biggest areas
    groove_mm: float = 0.5        # depth of the pencil lines
    rim_mm: float = 1.6           # outline rim above the backing
    rim_width_mm: float = 2.0
    line_threshold: float = 0.12  # how dark a stroke must be to count as a line


def _disk(r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return xx * xx + yy * yy <= r * r


def analyse(img: Image.Image, s: LineArtSettings):
    """Return (darkness 0..1, line mask, outer shape mask, region labels, is_figure dict)."""
    g = imaging.to_gray(img)
    k = max(3, int(min(g.shape) * 0.018)) | 1
    paper = ndimage.gaussian_filter(ndimage.median_filter(g, k), k)  # paper incl. shadows
    dark = np.clip(paper - g, 0, 1)
    lines = ndimage.binary_opening(dark > s.line_threshold, np.ones((2, 2)))

    filled = ndimage.binary_fill_holes(ndimage.binary_closing(lines, iterations=4))
    lab, n = ndimage.label(filled)
    if n == 0:
        raise ValueError("No closed outline found. Draw the outer shape as one closed line, "
                         "photograph it flat in good light.")
    sizes = ndimage.sum(filled, lab, range(1, n + 1))
    shape = lab == (int(np.argmax(sizes)) + 1)
    if shape.mean() < 0.02:
        raise ValueError("The drawing's outline is too small or not closed.")

    # Areas between the lines. Narrow necks (small gaps in the pencil line)
    # are cut by an opening, then pixels go to the nearest remaining seed.
    free = shape & ~ndimage.binary_dilation(lines, iterations=1)
    size = np.sqrt(shape.sum())
    seeds = ndimage.binary_opening(free, structure=_disk(max(2, int(size * 0.012))))
    slab, n = ndimage.label(seeds)
    if n:
        ssz = ndimage.sum(seeds, slab, range(1, n + 1))
        slab[np.isin(slab, np.flatnonzero(ssz < shape.sum() * 0.0005) + 1)] = 0
    if not slab.any():
        slab = ndimage.label(free)[0]
    _, (iy, ix) = ndimage.distance_transform_edt(slab == 0, return_indices=True)
    regions = np.where(free, slab[iy, ix], 0)

    # Background = areas that reach the outer outline.
    edge_d = ndimage.distance_transform_edt(shape)
    ids = [int(i) for i in np.unique(regions) if i]
    mind = ndimage.minimum(edge_d, regions, ids) if ids else []
    thr = size * 0.035
    is_figure = {i: bool(m >= thr) for i, m in zip(ids, mind)}
    return dark, lines, shape, regions, is_figure


def build(img: Image.Image, s: LineArtSettings, log: list[str]):
    """Return (height map mm, solid mask, number of figure / background areas)."""
    # 1) find the outline at a working size and crop the paper away
    work = imaging.limit_size(img, 900)
    _, _, shape0, _, _ = analyse(work, s)
    ys, xs = np.nonzero(shape0)
    f = img.width / work.width
    pad = 6
    box = (max(0, int((xs.min() - pad) * f)), max(0, int((ys.min() - pad) * f)),
           min(img.width, int((xs.max() + pad) * f)), min(img.height, int((ys.max() + pad) * f)))
    img = img.crop(box)
    # 2) work directly at the print resolution
    scale = (s.size_mm / s.pixel_mm) / max(img.size)
    img = img.resize((max(2, int(img.width * scale)), max(2, int(img.height * scale))),
                     Image.LANCZOS)
    dark, lines, shape, regions, is_figure = analyse(img, s)
    # Smooth the hand-drawn outline so the plaque edge is clean.
    shape = ndimage.gaussian_filter(shape.astype(float), 0.8 / s.pixel_mm) > 0.5
    regions = np.where(shape, regions, 0)
    n_fig = sum(is_figure.values())
    n_bg = len(is_figure) - n_fig
    log.append(f"Line art: {len(is_figure)} areas - {n_fig} figure, {n_bg} background")
    if n_fig == 0:
        log.append("No separate figure found - everything is one layer.")

    px = s.pixel_mm
    level = np.zeros(regions.shape)
    puff = np.zeros(regions.shape)
    objs = ndimage.find_objects(regions)
    biggest_r = 1e-6
    radii = {}
    for i, sl in enumerate(objs, start=1):
        if sl is None or i not in is_figure:
            continue
        m = regions[sl] == i
        d = ndimage.distance_transform_edt(m)
        # Soften the medial "creases" of the distance field -> rounder pillows.
        d = ndimage.gaussian_filter(d, max(1.0, d.max() / 4)) * m
        radii[i] = (sl, m, d)
        biggest_r = max(biggest_r, d.max())
    for i, (sl, m, d) in radii.items():
        r = max(d.max(), 1e-6)
        t = np.clip(d / r, 0, 1)
        profile = np.sqrt(np.clip(t * (2 - t), 0, 1))       # round pillow
        amount = s.puff_mm * np.sqrt(r / biggest_r)          # bigger area -> puffier
        if not is_figure[i]:
            amount *= 0.35                                    # background stays calm
        base = s.figure_mm if is_figure[i] else s.background_mm
        level[sl][m] = base
        puff[sl][m] = profile[m] * amount

    height = level + puff
    # Lines take the height of the nearest area, so a figure outline becomes a step.
    filled_px = regions > 0
    if filled_px.any():
        _, (iy, ix) = ndimage.distance_transform_edt(~filled_px, return_indices=True)
        height = height[iy, ix]
    height = ndimage.gaussian_filter(height, 0.8)
    # Pencil strokes as grooves (continuous: darker = deeper, shading shows).
    height -= np.clip(dark / 0.35, 0, 1) * s.groove_mm

    # Raised rim along the outline.
    edge_d = ndimage.distance_transform_edt(shape) * px
    rim = np.clip(1 - (edge_d - s.rim_width_mm) / 0.6, 0, 1)
    height = np.maximum(height, rim * s.rim_mm)

    height = s.base_mm + np.clip(height, 0, None)
    solid = ndimage.binary_dilation(shape, iterations=1)
    height = np.where(solid, height, 0.0)
    return height, solid, img
