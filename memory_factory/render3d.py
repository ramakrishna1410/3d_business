"""Tiny software renderer for product mockup images (no GPU / OpenGL needed).

Perspective camera, back-face culling, painter's algorithm, two-light
Lambert shading with a specular touch, supersampled for smooth edges.
Good enough for catalogue/WhatsApp mockups of our meshes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Item:
    vertices: np.ndarray
    faces: np.ndarray
    color: tuple
    shine: float = 0.3
    layer: int = 1   # lower layers are drawn first (e.g. 0 = table/board)


def _look_at(eye, target, up=(0, 0, 1)):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, -f]), eye


def render(items: list[Item], eye, target, size=(1600, 1000), fov_deg=35.0,
           background=((246, 241, 232), (222, 214, 200)), ss=2) -> Image.Image:
    W, H = size[0] * ss, size[1] * ss
    rot, eye = _look_at(eye, target)
    fpx = 0.5 * H / np.tan(np.radians(fov_deg) / 2)
    lights = [(np.array([-0.5, -0.8, 0.9]), 0.75), (np.array([0.8, -0.3, 0.4]), 0.35)]
    lights = [(d / np.linalg.norm(d), k) for d, k in lights]

    polys, depths, cols, layers = [], [], [], []
    for it in items:
        tri = it.vertices[it.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
        cen = tri.mean(axis=1)
        view = eye - cen
        facing = np.einsum("ij,ij->i", n, view) > 0
        tri, n, cen = tri[facing], n[facing], cen[facing]
        cam = (tri - eye) @ rot.T                      # camera space, looking down -z
        z = -cam[..., 2]
        ok = (z > 1).all(axis=1)
        cam, z, n, cen = cam[ok], z[ok], n[ok], cen[ok]
        sx = W / 2 + fpx * cam[..., 0] / z
        sy = H / 2 - fpx * cam[..., 1] / z
        shade = np.full(len(n), 0.22)
        for d, k in lights:
            shade += k * np.clip(n @ d, 0, 1)
        vdir = eye - cen
        vdir /= np.linalg.norm(vdir, axis=1, keepdims=True)
        h = lights[0][0] + vdir
        h /= np.linalg.norm(h, axis=1, keepdims=True)
        spec = np.clip(np.einsum("ij,ij->i", n, h), 0, 1) ** 30 * it.shine
        base = np.array(it.color, float)
        c = np.clip(base[None] * shade[:, None] + 255 * spec[:, None], 0, 255).astype(np.uint8)
        polys.append(np.stack([sx, sy], axis=-1))
        depths.append(z.mean(axis=1))
        cols.append(c)
        layers.append(np.full(len(c), it.layer))

    P = np.concatenate(polys)
    D = np.concatenate(depths)
    C = np.concatenate(cols)
    L = np.concatenate(layers)
    order = np.lexsort((-D, L))  # by layer, then far-to-near

    top, bottom = np.array(background[0], float), np.array(background[1], float)
    grad = np.linspace(0, 1, H)[:, None, None]
    bg = (top * (1 - grad) + bottom * grad).astype(np.uint8)
    img = Image.fromarray(np.broadcast_to(bg, (H, W, 3)).copy())
    draw = ImageDraw.Draw(img)
    for i in order:
        p = P[i]
        col = tuple(int(v) for v in C[i])
        draw.polygon([(p[0, 0], p[0, 1]), (p[1, 0], p[1, 1]), (p[2, 0], p[2, 1])], fill=col,
                     outline=col)
    return img.resize(size, Image.LANCZOS)
