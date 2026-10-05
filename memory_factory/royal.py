"""Royal Chess: a person's 3D head -> King / Queen / Bishop chess piece.

Input is a 3D head (+shoulders) model, either generated automatically from a photo
with the Tripo API (providers/tripo.py) or a GLB/OBJ/STL file exported from the
Tripo app (use the "3D Print" template so the beard, eyes and hair are carved
into the shape, not only painted).

Pipeline
    1. load + orient   z-up, face looking to -y (auto-detected from the face
                       profile; a manual turn override is available)
    2. neck cut        the narrowest slice under the head, moved down to just
                       above the shoulders/shirt so a beard is kept
    3. solid head      AI models are not closed (thousands of loose bits): the
                       surface is rasterised into voxels, every slice filled
    4. regalia         fitted to the actual head shape at the hairline:
                         king   - crown with points, pearls, cap, orb + cross
                         queen  - tiara (front arc), pearl necklace
                         bishop - mitre (child / kids piece)
    5. royal bust      shoulders, collar, chain/necklace, classic chess pedestal
    6. one solid       marching cubes -> union -> simplified watertight STL

Everything is built at chess-set scale (King ~80 mm) in a fixed layout and then
scaled: piece scale (queen 0.94, bishop 0.86) x size (chess set 1.0, couple gift 1.4).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import costing, engrave, mesh as mesh_mod, orders, render3d
from .config import load_pricing

PIECES = {"king": ("King", 1.0), "queen": ("Queen", 0.94), "bishop": ("Bishop (child)", 0.86)}
SIZES = {"chess set (King ~80 mm)": 1.0, "couple gift (King ~113 mm)": 1.4}
QUALITY_MM = {"preview": 0.25, "final": 0.1}       # voxel size on the printed piece
TURNS = ["auto", "0°", "90°", "180°", "270°"]
STYLES = ["smooth statue", "classic royal"]   # smooth: clean crown, folded mantle, stand-up collar
FINISH_COLOURS = {"bronze": (176, 120, 70), "gold": (222, 178, 92), "statue white": (232, 230, 224),
                  "raw resin": (200, 200, 205)}

# layout at chess scale (mm)
HEAD_H = 24.65          # neck cut -> top of the head
Z_NECK = 47.0           # where the neck cut sits on the piece
PED_TOP = 34.0
TORSO = dict(A=19.0, B=8.0, C=12.0, n=2.6, yc=1.5)
BASE_R = 17.0
COLLAR = (8.2, 8.8, -0.2)   # neck opening of the bust: half-width x, half-depth y, centre y


@dataclass
class RoyalSettings:
    piece: str = "king"                               # king / queen / bishop
    style: str = "smooth statue"                      # see STYLES
    size: str = "couple gift (King ~113 mm)"
    quality: str = "preview"                          # preview / final
    turn: str = "auto"
    neck: float | None = None                         # neck cut, fraction of model height
    crown: float = 0.77                               # crown band, fraction of the head height
    tidy_hair: bool = True                            # trim loose strands / hair hanging below the neck
    name: str = ""                                    # engraved on the front of the stand
    message1: str = ""                                # engraved under the base
    message2: str = ""
    date: str = ""
    font: str = "Script (Great Vibes)"                # see engrave.FONTS
    finish: str = "bronze"
    packaging: str = "velvet box"

    @property
    def scale(self) -> float:
        return PIECES[self.piece][1] * SIZES.get(self.size, 1.0)


@dataclass
class Result:
    order_id: str
    folder: Path
    stl: Path
    glb: Path | None
    render: Path
    quote_md: str
    log: list[str]
    volume_mm3: float
    height_mm: float
    head_model: Path | None = None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- loading
def load_head(path) -> "object":
    import trimesh

    m = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
        raise ValueError("Could not read a 3D mesh from this file (use GLB, OBJ or STL).")
    return m


def _rot(axis, deg):
    import trimesh

    return trimesh.transformations.rotation_matrix(np.radians(deg), axis)[:3, :3]


def _profile_roughness(v: np.ndarray, d: np.ndarray) -> float:
    """Bumpiness of the centre-line profile seen from direction d (a face has
    nose, lips and chin; the back of a head is a smooth curve)."""
    z0, z1 = v[:, 2].min(), v[:, 2].max()
    h = z1 - z0
    c = v[:, :2].mean(axis=0)
    p = (v[:, :2] - c) @ d
    perp = (v[:, :2] - c) @ np.array([-d[1], d[0]])
    near = np.abs(perp) < 0.12 * h
    zs = np.linspace(z0 + 0.3 * h, z0 + 0.8 * h, 120)
    zi = np.clip(((v[:, 2] - zs[0]) / (zs[1] - zs[0])).round().astype(int), -1, len(zs))
    ok = near & (zi >= 0) & (zi < len(zs)) & (np.abs(v[:, 2] - zs[np.clip(zi, 0, len(zs) - 1)]) < h / 240)
    prof = np.full(len(zs), -np.inf)
    np.maximum.at(prof, zi[ok], p[ok])
    good = np.isfinite(prof)
    if good.sum() < 10:
        return 0.0
    prof = np.interp(np.arange(len(zs)), np.where(good)[0], prof[good])
    return float(np.abs(prof - ndimage.gaussian_filter1d(prof, 8)).sum() / h)


def orient(m, suffix: str, turn: str, log: list[str]) -> np.ndarray:
    """Return vertices z-up with the face looking towards -y."""
    v = np.asarray(m.vertices, float)
    if suffix.lower() in (".glb", ".gltf", ".obj", ".fbx"):   # y-up formats
        v = v @ _rot([1, 0, 0], 90).T
    ext = v.max(axis=0) - v.min(axis=0)
    if ext[2] < 0.75 * ext.max():                               # tallest axis is "up"
        ax = int(np.argmax(ext))
        v = v @ (_rot([0, 1, 0], -90) if ax == 0 else _rot([1, 0, 0], 90)).T
    # shoulders (wide) at the bottom, skull (narrow) at the top
    z0, z1 = v[:, 2].min(), v[:, 2].max()
    lo = v[v[:, 2] < z0 + 0.15 * (z1 - z0)]
    hi = v[v[:, 2] > z1 - 0.15 * (z1 - z0)]
    if np.ptp(hi[:, 0]) * np.ptp(hi[:, 1]) > 1.3 * np.ptp(lo[:, 0]) * np.ptp(lo[:, 1]):
        v = v @ _rot([1, 0, 0], 180).T
    if turn == "auto":
        dirs = {0: (0, -1), 90: (1, 0), 180: (0, 1), 270: (-1, 0)}  # face direction -> turn
        scores = {a: _profile_roughness(v, np.array(d, float)) for a, d in dirs.items()}
        deg = max(scores, key=scores.get)
        log.append(f"Face direction detected automatically (turned {deg}°)")
    else:
        deg = int(turn.rstrip("°"))
    # rotate so that the detected face direction ends up at -y
    return v @ _rot([0, 0, 1], -deg).T if deg else v


def landmarks(v: np.ndarray, neck: float | None) -> dict:
    z0, z1 = v[:, 2].min(), v[:, 2].max()
    h = z1 - z0
    zs = z0 + h * np.linspace(0.08, 0.5, 43)
    widths = []
    for z in zs:
        s = v[np.abs(v[:, 2] - z) < h * 0.006]
        widths.append(np.ptp(s[:, 0]) if len(s) > 2 else np.nan)
    widths = np.array(widths)
    if neck is not None:
        z_cut = z0 + float(neck) * h
    else:
        i_n = int(np.nanargmin(np.where(zs > z0 + 0.15 * h, widths, np.nan)))
        z_cut = zs[i_n] - 0.1 * h
        for j in range(i_n, -1, -1):          # walk down to the shoulders / shirt
            if widths[j] > 1.45 * widths[i_n]:
                z_cut = zs[j] + 0.01 * h
                break
    top = z1
    zm = z_cut + 0.55 * (top - z_cut)
    s = v[np.abs(v[:, 2] - zm) < h * 0.01]
    cx = (s[:, 0].min() + s[:, 0].max()) / 2
    cy = (s[:, 1].min() + s[:, 1].max()) / 2
    return dict(z0=z0, top=top, z_cut=z_cut, cx=cx, cy=cy, neck_frac=(z_cut - z0) / h)


# ---------------------------------------------------------------- voxel grid
class Grid:
    """Occupancy field on a regular grid; parts are added as signed distances (mm, + inside)."""

    def __init__(self, lo, hi, pitch):
        self.p = float(pitch)
        self.x = np.arange(lo[0], hi[0] + pitch / 2, pitch)
        self.y = np.arange(lo[1], hi[1] + pitch / 2, pitch)
        self.z = np.arange(lo[2], hi[2] + pitch / 2, pitch)
        self.f = np.zeros((len(self.x), len(self.y), len(self.z)), np.float32)

    def xyz(self):
        return np.meshgrid(self.x, self.y, self.z, indexing="ij", sparse=True)

    def add(self, sd):
        np.maximum(self.f, np.clip(0.5 + sd / self.p, 0, 1).astype(np.float32), out=self.f)

    def to_trimesh(self, blur=0.7):
        import trimesh
        from skimage import measure

        f = np.pad(ndimage.gaussian_filter(self.f, blur) if blur else self.f, 1)
        inside = f > 0.5
        f[ndimage.binary_fill_holes(inside) & ~inside] = 1.0   # no sealed air pockets (trap resin)
        f[np.abs(f - 0.5) < 1e-4] = 0.5 + 1e-4      # exact iso-values make non-manifold edges
        v, fc, _, _ = measure.marching_cubes(f, 0.5, spacing=(self.p,) * 3)
        v += [self.x[0] - self.p, self.y[0] - self.p, self.z[0] - self.p]
        return trimesh.Trimesh(v, fc[:, ::-1], process=False)


def _sphere(g, c, r):
    X, Y, Z = g.xyz()
    return r - np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2)


def _box(g, size, c):
    X, Y, Z = g.xyz()
    return np.minimum(np.minimum(size[0] / 2 - np.abs(X - c[0]), size[1] / 2 - np.abs(Y - c[1])),
                      size[2] / 2 - np.abs(Z - c[2]))


def _capsule(g, p0, p1, r):
    X, Y, Z = g.xyz()
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    t = np.clip(((X - p0[0]) * d[0] + (Y - p0[1]) * d[1] + (Z - p0[2]) * d[2]) / (d @ d), 0, 1)
    return r - np.sqrt((X - p0[0] - t * d[0]) ** 2 + (Y - p0[1] - t * d[1]) ** 2 + (Z - p0[2] - t * d[2]) ** 2)


def _ring(g, c, ax, ay, r_tube, tilt=0.0):
    X, Y, Z = g.xyz()
    dr = (np.hypot(X / ax, (Y - c[1]) / ay) - 1) * (ax + ay) / 2
    return r_tube - np.hypot(dr, Z - (c[2] + tilt * (Y - c[1])))


def _revolve(g, prof):
    X, Y, Z = g.xyz()
    prof = np.asarray(prof, float)
    R = np.interp(g.z, prof[:, 1], prof[:, 0], left=-1, right=-1)[None, None, :]
    return np.minimum(R - np.hypot(X, Y), np.minimum(Z - prof[:, 1].min(), prof[:, 1].max() - Z))


# ---------------------------------------------------------------- head
def solid_head(v, faces, pitch, lo, hi, z_cut, log) -> tuple[np.ndarray, Grid]:
    import trimesh

    vs, fs = trimesh.remesh.subdivide_to_size(v, faces, max_edge=pitch * 0.9, max_iter=10)
    g = Grid(lo, hi, pitch)
    shape = g.f.shape
    idx = np.round((vs - lo) / pitch).astype(int)
    ok = np.all((idx >= 0) & (idx < shape), axis=1)
    idx = idx[ok]
    shell = np.zeros(shape, bool)
    shell[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    del vs, fs, idx
    shell = ndimage.binary_dilation(shell, iterations=1)     # close pin holes
    solid = np.zeros_like(shell)
    for k in range(shape[2]):                                # open bottom: fill slice by slice
        solid[:, :, k] = ndimage.binary_fill_holes(shell[:, :, k])
    solid = ndimage.binary_fill_holes(solid)
    solid = ndimage.binary_erosion(solid, iterations=1)
    solid[:, :, g.z < z_cut] = False
    lab, n = ndimage.label(solid)
    if n == 0:
        raise ValueError("Could not build a solid head from this model.")
    sizes = ndimage.sum(solid, lab, range(1, n + 1))
    solid = lab == (int(np.argmax(sizes)) + 1)
    log.append(f"Solid head: {int(solid.sum()):,} voxels of {pitch:.2f} mm ({n} loose parts merged/removed)")
    return solid, g


def tidy_hair(solid, g, log) -> np.ndarray:
    """Make long or loose hair printable: hair hanging below the jaw is shaped into a smooth
    taper that ends inside the collar, and thin loose strands (fragile in resin) are removed.
    The face and beard (front) are never touched."""
    ax, ay, yc = COLLAR[0] + 0.4, COLLAR[1] + 0.4, COLLAR[2]
    k_top = int(np.searchsorted(g.z, Z_NECK + 0.5 * HEAD_H))
    back = (g.y > -3.0)[None, :]                         # behind the front of the neck
    n0 = int(solid.sum())
    for k in range(k_top):
        grow = 1 + 0.09 * max(0.0, g.z[k] - Z_NECK)
        out = np.hypot(g.x[:, None] / (ax * grow), (g.y[None, :] - yc) / (ay * grow)) > 1
        solid[:, :, k] &= ~(out & back)
    r = max(1, int(round(0.3 / g.p)))
    blk = solid[:, :, :k_top]
    ball = ndimage.generate_binary_structure(3, 1)
    opened = ndimage.binary_opening(blk, ball, iterations=r)
    solid[:, :, :k_top] = np.where(back[:, :, None], opened, blk)
    lab, n = ndimage.label(solid)
    if n > 1:
        solid = lab == (int(np.argmax(ndimage.sum(solid, lab, range(1, n + 1)))) + 1)
    log.append(f"Tidy hair: {100 * (1 - solid.sum() / max(n0, 1)):.1f}% of the head trimmed "
               "(loose strands, hair below the neck)")
    return solid


def head_field(solid, g, v, faces, log, layers: int = 4, smooth: float = 1.5) -> np.ndarray:
    """Occupancy field whose 0.5 level follows the *original* head surface (sub-voxel), so the
    face comes out smooth instead of stepped. The voxel solid still decides what is inside
    (holes filled, neck cut, tidy hair); near the surface the distance to the Tripo surface is
    used, from a moving-least-squares fit of its points and normals. Works in chunks (low RAM)."""
    import trimesh
    from scipy.spatial import cKDTree

    p = g.p
    # ring index: -layers..layers voxels from the solid boundary (+ inside), cheap boolean ops
    ring = np.where(solid, layers + 1, -(layers + 1)).astype(np.int8)
    cur = solid.copy()
    for i in range(layers, 0, -1):            # inside rings
        nxt = ndimage.binary_erosion(cur)
        ring[cur & ~nxt] = layers + 1 - i
        cur = nxt
    cur = solid.copy()
    for i in range(1, layers + 1):            # outside rings
        nxt = ndimage.binary_dilation(cur)
        ring[nxt & ~cur] = -i
        cur = nxt
    del cur, nxt
    f = solid.astype(np.float32)

    vs, fs = trimesh.remesh.subdivide_to_size(v, faces, max_edge=p, max_iter=12)
    vn = np.asarray(trimesh.Trimesh(vs, fs, process=False).vertex_normals, np.float32)
    del fs
    vi = np.round((vs - [g.x[0], g.y[0], g.z[0]]) / p).astype(int)
    okk = np.all((vi >= 0) & (vi < ring.shape), axis=1)
    r_at = np.full(len(vs), 99, np.int16)
    r_at[okk] = ring[vi[okk, 0], vi[okk, 1], vi[okk, 2]]
    keep = np.abs(r_at) <= 2                  # visible outer skin only (not eyeballs, inner shells)
    vs, vn = vs[keep].astype(np.float32), vn[keep]

    def ring_at(q):
        qi = np.clip(np.round((q - [g.x[0], g.y[0], g.z[0]]) / p).astype(int), 0, np.array(ring.shape) - 1)
        return ring[qi[:, 0], qi[:, 1], qi[:, 2]]
    flip = ring_at(vs + vn * 2 * p) > ring_at(vs - vn * 2 * p)     # outward = towards outside
    vn[flip] *= -1
    tree = cKDTree(vs)

    band = np.abs(ring) <= layers
    ii = np.nonzero(band)
    h = smooth * p
    for a in range(0, len(ii[0]), 400_000):
        sl = slice(a, a + 400_000)
        P = np.column_stack([g.x[ii[0][sl]], g.y[ii[1][sl]], g.z[ii[2][sl]]]).astype(np.float32)
        d, j = tree.query(P, k=8, workers=-1)
        w = np.exp(-(d / h) ** 2)
        w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
        n = (w[..., None] * vn[j]).sum(1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        c = (w[..., None] * vs[j]).sum(1)
        sd = -np.einsum("nc,nc->n", P - c, n)                    # + inside
        e = ring[ii[0][sl], ii[1][sl], ii[2][sl]].astype(np.float32)
        e = (e - np.sign(e) * 0.5) * p                            # voxel-based distance
        sd = np.where(d[:, 0] > 2.5 * p, e, np.minimum(sd, e + 2.5 * p))   # never outside the solid
        f[ii[0][sl], ii[1][sl], ii[2][sl]] = np.clip(0.5 + sd / p, 0, 1)
    log.append(f"Smooth face: surface fitted to {len(vs):,} points of the original model")
    return f


def _section(solid, g, z_band, depth_mm=2.0):
    """Smoothed horizontal section of the head around the band height."""
    kb = int(np.argmin(np.abs(g.z - z_band)))
    k1 = min(solid.shape[2], kb + max(1, int(depth_mm / g.p)))
    sec = solid[:, :, max(0, kb - 2):k1].any(axis=2)
    sec = ndimage.gaussian_filter(sec.astype(float), 1.0 / g.p) > 0.35
    sec = ndimage.binary_fill_holes(sec)
    if not sec.any():
        raise ValueError("The crown height is above the head - lower the crown slider.")
    dout = ndimage.distance_transform_edt(~sec) * g.p
    din = ndimage.distance_transform_edt(sec) * g.p
    ix, iy = np.nonzero(sec)
    return sec, dout, din, g.x[int(round(ix.mean()))], g.y[int(round(iy.mean()))]


def _point_at(g, dout, cx, cy, ang, off):
    """Walk out from the centre at angle `ang` until `off` mm outside the head section."""
    dx, dy = np.sin(ang), -np.cos(ang)
    for rr in np.arange(1.0, 25.0, g.p / 2):
        ix = int(round((cx + dx * rr - g.x[0]) / g.p))
        iy = int(round((cy + dy * rr - g.y[0]) / g.p))
        if not (0 <= ix < dout.shape[0] and 0 <= iy < dout.shape[1]) or dout[ix, iy] >= off:
            break
    return cx + dx * rr, cy + dy * rr


def add_crown(g, solid, z_band):
    sec, dout, din, cx, cy = _section(solid, g, z_band, 3.4)
    X, Y, Z = g.xyz()
    hb, t, gap = 3.4, 1.0, 0.25
    th = np.arctan2(X - cx, -(Y - cy))
    tri = np.clip(1 - 2 * np.abs((th * 8 / (2 * np.pi) + 0.5) % 1.0 - 0.5), 0, 1)
    d2 = (dout - din)[:, :, None]            # signed: negative inside the head section
    ztop = z_band + hb + 3.0 * tri ** 1.6
    # the band reaches 1.2 mm into the head so it is fused to it (one solid)
    g.add(np.minimum(np.minimum(d2 + 1.2, gap + t - d2), np.minimum(Z - z_band, ztop - Z)))
    g.add(np.minimum(0.65 - np.abs(d2 - (gap + t * 0.6)), 0.65 - np.abs(Z - (z_band + 0.45))))
    capz = z_band + hb - 0.5 + 4.6 * np.sqrt(np.clip(din / max(din.max(), 1e-6), 0, 1))[:, :, None]
    inside = np.where(sec, 1.0, -1.0)[:, :, None]
    g.add(np.minimum(np.minimum(capz - Z, inside), Z - (z_band + 1.0)))
    top = z_band + hb - 0.5 + 4.6
    g.add(_sphere(g, (cx, cy, top + 0.9), 1.7))
    g.add(_box(g, (1.4, 1.4, 5.0), (cx, cy, top + 4.4)))
    g.add(_box(g, (4.0, 1.4, 1.4), (cx, cy, top + 5.0)))
    for k in range(8):
        a = k * 2 * np.pi / 8
        px, py = _point_at(g, dout, cx, cy, a, gap + t * 0.5)
        g.add(_sphere(g, (px, py, z_band + hb + 3.3), 0.7))
        px, py = _point_at(g, dout, cx, cy, a + np.pi / 8, gap + t + 0.15)
        g.add(_sphere(g, (px, py, z_band + 1.8), 0.65))


def add_crown_smooth(g, solid, z_band):
    """Statue crown: smooth band with rims, tall ball-tipped points, orb and cross pattee."""
    sec, dout, din, cx, cy = _section(solid, g, z_band, 3.0)
    X, Y, Z = g.xyz()
    hb, t, ph = 3.0, 1.15, 4.4
    th = np.arctan2(X - cx, -(Y - cy))
    tri = np.clip(1 - 2 * np.abs((th * 8 / (2 * np.pi) + 0.5) % 1.0 - 0.5), 0, 1)
    d2 = (dout - din)[:, :, None]
    ztop = z_band + hb + ph * tri ** 1.15
    g.add(np.minimum(np.minimum(d2 + 1.2, 0.25 + t - d2), np.minimum(Z - z_band, ztop - Z)))
    for zr in (z_band + 0.45, z_band + hb - 0.35):                     # rim mouldings
        g.add(np.minimum(0.55 - np.abs(d2 - (0.25 + t)), 0.5 - np.abs(Z - zr)))
    capz = z_band + hb + 1.8 * np.sqrt(np.clip(din / max(din.max(), 1e-6), 0, 1))[:, :, None]
    inside = np.where(sec, 1.0, -1.0)[:, :, None]
    g.add(np.minimum(np.minimum(capz - Z, inside), Z - (z_band + 1.0)))
    for k in range(8):                                                   # balls on the points
        px, py = _point_at(g, dout, cx, cy, k * 2 * np.pi / 8, 0.25 + t * 0.5)
        g.add(_sphere(g, (px, py, z_band + hb + ph + 0.55), 1.05))
    top = z_band + hb + 1.8
    g.add(_sphere(g, (cx, cy, top + 1.0), 1.6))                          # orb
    zc = top + 5.2                                                       # cross pattee
    g.add(_box(g, (1.5, 1.5, 6.2), (cx, cy, zc)))
    g.add(_box(g, (5.6, 1.5, 1.5), (cx, cy, zc + 0.6)))
    for dx, dz, w, h in ((0, 3.1, 2.6, 0.9), (0, -1.9, 2.4, 0.9), (2.8, 0.6, 0.9, 2.6), (-2.8, 0.6, 0.9, 2.6)):
        g.add(_box(g, (w, 1.5, h), (cx + dx, cy, zc + dz)))


def add_tiara_smooth(g, solid, z_band):
    """Statue coronet: full smooth band, 12 rounded points (tall/short) with pearls,
    a raised front peak and a jewel at the front."""
    sec, dout, din, cx, cy = _section(solid, g, z_band, 2.5)
    X, Y, Z = g.xyz()
    hb, t = 1.9, 1.0
    th = np.arctan2(X - cx, -(Y - cy))
    w = 0.5 + 0.5 * np.cos(12 * th)                                   # 12 points, one at the front
    tall = 0.5 + 0.5 * np.cos(6 * th)                                  # alternate tall / short
    peak = np.clip(np.cos(th), 0, 1) ** 14                             # front centre
    ztop = z_band + hb + (1.3 + 1.7 * tall + 2.0 * peak) * w ** 2.5
    d2 = (dout - din)[:, :, None]
    g.add(np.minimum(np.minimum(d2 + 1.2, 0.2 + t - d2), np.minimum(Z - z_band, ztop - Z)))
    g.add(np.minimum(0.5 - np.abs(d2 - (0.2 + t)), 0.45 - np.abs(Z - (z_band + 0.4))))   # rim
    for k in range(12):
        a = k * 2 * np.pi / 12
        h = 1.3 + (1.7 if k % 2 == 0 else 0.0) + (2.0 if k == 0 else 0.0)
        r = 1.0 if k == 0 else (0.7 if k % 2 == 0 else 0.55)
        px, py = _point_at(g, dout, cx, cy, a, 0.2 + t * 0.5)
        g.add(_sphere(g, (px, py, z_band + hb + h + r * 0.5), r))
    px, py = _point_at(g, dout, cx, cy, 0.0, 0.2 + t)
    g.add(_sphere(g, (px, py - 0.1, z_band + hb * 0.55), 0.85))           # jewel on the front


def add_tiara(g, solid, z_band):
    sec, dout, din, cx, cy = _section(solid, g, z_band, 2.0)
    X, Y, Z = g.xyz()
    t, gap = 0.8, 0.2
    th = np.arctan2(X - cx, -(Y - cy))
    front = np.clip(np.cos(th), 0, 1)
    sc = np.clip(1 - 2 * np.abs(((th * 11 / (2 * np.pi)) + 0.5) % 1 - 0.5), 0, 1)
    d2 = (dout - din)[:, :, None]
    z0 = z_band - 0.6 * front      # sits a little lower at the forehead
    ztop = z0 + 0.9 + front ** 2 * (2.8 + 1.0 * sc ** 2)
    sd = np.minimum(np.minimum(d2 + 1.2, gap + t - d2), np.minimum(Z - z0, ztop - Z))
    g.add(np.minimum(sd, np.cos(th) + 0.25))                    # front arc only, fused to the hair
    for k in range(-4, 5):
        a = k * 2 * np.pi / 11
        px, py = _point_at(g, dout, cx, cy, a, gap + t * 0.5)
        fr = max(0.0, np.cos(a))
        g.add(_sphere(g, (px, py, z_band - 0.6 * fr + 0.9 + fr ** 2 * 3.8 + 0.4), 0.55 if k else 1.0))


def add_mitre(g, solid, z_band):
    sec, dout, din, cx, cy = _section(solid, g, z_band, 1.0)
    X, Y, Z = g.xyz()
    ix, iy = np.nonzero(sec)
    a0 = (g.x[ix].max() - g.x[ix].min()) / 2 + 0.7
    b0 = (g.y[iy].max() - g.y[iy].min()) / 2 + 0.7
    H = 14.0
    u = np.clip((Z - z_band) / H, 0, 1)
    ax = a0 * np.clip(1 - u, 0, 1) ** 0.6 * (1 + 0.25 * u * (1 - u))   # pointed arch seen from the front
    ay = b0 * np.clip(1 - u ** 1.5, 0, 1) ** 0.8
    q = np.sqrt(((X - cx) / np.maximum(ax, 1e-3)) ** 2 + ((Y - cy) / np.maximum(ay, 1e-3)) ** 2)
    sd = np.minimum((1 - q) * np.minimum(ax, ay), np.minimum(Z - z_band, z_band + H - Z))
    sd = np.minimum(sd, np.maximum(np.abs(Y - cy) - 0.45, 0.55 - u) * 1.0)   # cleft between the peaks
    g.add(sd)
    g.add(_ring(g, (cx, cy, z_band + 0.6), a0 + 0.15, b0 + 0.15, 0.7))       # band
    yf = cy - b0 * 0.92                                                         # cross on the front
    g.add(_box(g, (0.9, 1.4, 5.2), (cx, yf + 0.9, z_band + 4.6)))
    g.add(_box(g, (3.4, 1.4, 0.9), (cx, yf + 0.9, z_band + 5.6)))


# ---------------------------------------------------------------- bust + pedestal
def body(piece: str, pitch: float, style: str = "smooth statue") -> object:
    A, B, C, n, yc0 = (TORSO[k] for k in ("A", "B", "C", "n", "yc"))
    g = Grid((-A - 2, -19, PED_TOP - 4), (A + 2, 19, Z_NECK + 6), pitch)   # pedestal: see stand()
    X, Y, Z = g.xyz()
    smooth = style == "smooth statue"
    zb = PED_TOP - 0.5
    zc = zb + 3.5
    q = (np.abs(X / A) ** n + np.abs((Y - yc0) / B) ** n + np.abs((Z - zc) / C) ** n) ** (1 / n)
    sd = np.minimum((1 - q) * B, Z - (zb + 0.04 * X ** 2))       # classical bust truncation
    zn = Z_NECK + 0.6
    if smooth and piece == "queen":
        # gown: smooth bodice with a rounded neckline and piping; bare chest above it
        front = np.clip(-(Y - yc0) / B * 1.6, 0, 1)
        zline = zn - 2.6 - 6.0 * front * np.clip(1 - (X / 10.5) ** 2, 0, 1)
        bodice = np.minimum(sd + 0.45, zline - Z)
        piping = np.minimum(sd + 0.85, 0.5 - np.abs(Z - zline))
        sd = np.maximum(sd, np.maximum(bodice, piping))
    elif smooth:
        # mantle draped over the shoulders: folds fanning out from the collar, smooth chest panel
        front = np.clip(-(Y - yc0) / B, 0, 1)
        phi = np.arctan2(X, (zc + C + 3.0) - Z)
        rr = np.hypot(X, (zc + C + 3.0) - Z)
        fade = np.clip((rr - 8.0) / 5.0, 0, 1)                         # folds start below the collar
        folds = 0.6 * fade * (0.5 + 0.5 * np.cos(9 * phi)) ** 1.5      # a few broad, soft folds
        panel = np.clip((np.abs(X) - 5.2) / 0.8, 0, 1)                 # no folds on the chest panel
        lapel = np.clip(0.75 - np.abs(np.abs(X) - 5.6) * 1.2, 0, 0.75) * front
        placket = np.clip(0.45 - np.abs(np.abs(X) - 1.1) * 1.5, 0, 0.45) * front
        sd = np.minimum(sd + folds * panel + lapel + placket * (Z > zb + 2),
                        Z - (zb + 0.04 * X ** 2))
    g.add(np.minimum(sd, (A - 1.5) - np.abs(X)))

    def chest_y(x, z):
        r = 1 - np.abs(x / A) ** n - np.abs((z - zc) / C) ** n
        return yc0 - B * np.clip(r, 0, 1) ** (1 / n)

    g.add(_capsule(g, (0, 1.6, zc + 4), (0, 1.0, Z_NECK + 6), 5.4))          # neck plug
    if smooth and piece == "queen":
        ax, ay, yc = COLLAR
        rho = np.hypot(X / ax, (Y - yc) / ay)
        dr = (rho - 1) * (ax + ay) / 2
        g.add(np.minimum(np.minimum(-dr, Z_NECK + 0.6 - Z), Z - (Z_NECK - 4)))   # closes the neck cut
        g.add(_ring(g, (0, yc, Z_NECK + 0.3), ax, ay, 0.9, tilt=0.1))             # low band
        for tt in np.linspace(0, 2 * np.pi, 28, endpoint=False):                 # pearl choker on it
            y = yc - ay * np.cos(tt)
            g.add(_sphere(g, (ax * np.sin(tt), y, Z_NECK + 1.3 + 0.1 * (y - yc)), 0.95))
        for tt in np.linspace(-1.05, 1.05, 15):                                   # pearl necklace
            x, z = 7.6 * np.sin(tt), zn - 2.8 - 3.6 * np.cos(tt) ** 2
            g.add(_sphere(g, (x, chest_y(x, z) + 0.2, z), 0.7))
        zp = zn - 2.8 - 3.6 - 1.6
        g.add(_sphere(g, (0, chest_y(0, zp) + 0.05, zp), 1.15))                  # drop pearl
    elif smooth:                   # stand-up collar (hides the neck cut), open at the front
        ax, ay, t = COLLAR[0], COLLAR[1], 1.3
        rho = np.hypot(X / ax, (Y + 0.2) / ay)
        dr = (rho - 1) * (ax + ay) / 2
        zt = Z_NECK + 2.8 + 0.12 * (Y + 0.2)                         # a little higher at the back
        shell = np.minimum(t / 2 - np.abs(dr + t / 2), np.minimum(Z - (Z_NECK - 3.5), zt - Z))
        notch = np.maximum(np.abs(X) - (0.25 + 0.45 * (Z - Z_NECK)), Y + 4)   # small V at the front
        g.add(np.minimum(shell, np.maximum(notch, Z_NECK - 0.5 - Z)))
        g.add(_ring(g, (0, -0.2, Z_NECK + 2.6), ax - 0.2, ay - 0.2, 0.75, tilt=0.12))   # rolled top edge
        # fill between the neck and the collar so there is no gap (or resin trap) inside it
        g.add(np.minimum(np.minimum(-dr, Z_NECK + 1.0 - Z), Z - (Z_NECK - 3.5)))
        if piece == "bishop":      # small cross on the chest
            zx = zn - 9.0
            yx = chest_y(0, zx) - 0.3
            g.add(_box(g, (1.0, 1.2, 5.0), (0, yx, zx)))
            g.add(_box(g, (3.4, 1.2, 1.0), (0, yx, zx + 0.9)))
    elif piece == "king":          # ermine collar + chain of office with a medallion
        g.add(_ring(g, (0, 0.6, zn), 8.6, 8.0, 2.4, tilt=0.18))
        for tt in np.linspace(-1.2, 1.2, 25):
            x, z = 12.5 * np.sin(tt), zn - 3.0 - 6.5 * np.cos(tt) ** 2
            g.add(_sphere(g, (x, chest_y(x, z) + 0.15, z), 0.65))
        zm = zn - 12.0
        yc = chest_y(0, zm)
        g.add(np.minimum(_sphere(g, (0, yc + 0.2, zm), 2.7), 0.75 - np.abs(Y - (yc - 0.45))))
    elif piece == "queen":         # pearl-studded collar + pearl necklace with a pendant
        g.add(_ring(g, (0, 0.6, zn), 8.2, 7.6, 1.6, tilt=0.18))
        for tt in np.linspace(0, 2 * np.pi, 22, endpoint=False):
            g.add(_sphere(g, (9.6 * np.sin(tt), 0.6 - 9.0 * np.cos(tt), zn + 0.18 * -9.0 * np.cos(tt)), 0.7))
        for tt in np.linspace(-1.15, 1.15, 21):
            x, z = 10.0 * np.sin(tt), zn - 2.5 - 5.0 * np.cos(tt) ** 2
            g.add(_sphere(g, (x, chest_y(x, z) + 0.2, z), 0.75))
        zp = zn - 2.5 - 5.0 - 1.8
        g.add(_sphere(g, (0, chest_y(0, zp) - 0.1, zp), 1.3))
    else:                          # bishop: simple rolled collar + small cross
        g.add(_ring(g, (0, 0.6, zn), 8.2, 7.6, 1.9, tilt=0.18))
        zx = zn - 8.0
        yx = chest_y(0, zx) - 0.3
        g.add(_box(g, (1.0, 1.2, 5.0), (0, yx, zx)))
        g.add(_box(g, (3.4, 1.2, 1.0), (0, yx, zx + 0.9)))
    return g.to_trimesh()


# ---------------------------------------------------------------- stand (turned, perfectly smooth)
def _stand_profile(style: str):
    R = BASE_R
    if style == "smooth statue":   # stepped base with rounded rings, slim stem, double ring under the bust
        prof = [(0, 0), (R, 0), (R, 2.4), (0.95 * R, 3.0), (0.95 * R, 4.4), (0.86 * R, 5.0),
                (0.86 * R, 6.6), (0.72 * R, 7.4), (0.68 * R, 9.0), (0.55 * R, 10.0), (0.42 * R, 12.5),
                (0.33 * R, PED_TOP - 9), (0.36 * R, PED_TOP - 5.5), (0.46 * R, PED_TOP - 4.4),
                (0.46 * R, PED_TOP - 1.0), (0.3 * R, PED_TOP + 2), (0.0, PED_TOP + 2)]
        rings = ((0.97 * R, 1.6, 1.0), (0.9 * R, 4.7, 0.85), (0.72 * R, 8.0, 0.8),
                 (0.47 * R, PED_TOP - 3.4, 0.95), (0.48 * R, PED_TOP - 1.0, 0.95))
        stem = ((0.42 * R, 12.5), (0.33 * R, PED_TOP - 9))
    else:
        prof = [(0, 0), (R, 0), (R, 3), (0.92 * R, 4.5), (0.85 * R, 6.5), (0.62 * R, 8.5), (0.42 * R, 12),
                (0.33 * R, PED_TOP - 9), (0.36 * R, PED_TOP - 4), (0.46 * R, PED_TOP - 2.5),
                (0.46 * R, PED_TOP - 1), (0.3 * R, PED_TOP + 2), (0.0, PED_TOP + 2)]
        rings = ()
        stem = ((0.42 * R, 12), (0.33 * R, PED_TOP - 9))
    return np.array(prof, float), rings, np.array(stem, float)


def _round_corners(p: np.ndarray, r: float = 0.5, steps: int = 5) -> np.ndarray:
    """Small fillets on every corner of the lathe profile (ends on the axis kept)."""
    out = [p[0]]
    for a, b, c in zip(p[:-2], p[1:-1], p[2:]):
        ta = min(r / max(np.linalg.norm(a - b), 1e-9), 0.45)
        tc = min(r / max(np.linalg.norm(c - b), 1e-9), 0.45)
        p0, p2 = b + (a - b) * ta, b + (c - b) * tc
        for t in np.linspace(0, 1, steps):
            out.append((1 - t) ** 2 * p0 + 2 * (1 - t) * t * b + t ** 2 * p2)
    out.append(p[-1])
    q = np.array(out)
    q[:, 0] = np.maximum(q[:, 0], 0.0)
    return q


def stand(style: str):
    """The pedestal as a lathe-turned solid (layout units) - smooth surface, no voxel steps."""
    import manifold3d as mf

    prof, rings, _ = _stand_profile(style)
    st = mf.Manifold.revolve(mf.CrossSection([_round_corners(prof)]), 256)
    for rr, zr, tube in rings:
        st = st + mf.Manifold.revolve(mf.CrossSection.circle(tube, 48).translate([rr, zr]), 256)
    return st


def _plan_text(s: RoyalSettings) -> dict:
    """Fit the name / message / date to this piece and size (fast, no geometry). Raises a
    readable ValueError when something does not fit."""
    plan = {}
    if s.font not in engrave.FONTS:
        raise ValueError(f"Unknown font {s.font!r}")
    k = s.scale
    _, _, stem = _stand_profile(s.style)
    stem = stem * k
    if s.name.strip():
        z_mid = stem.mean(axis=0)[1] + 0.6 * k
        r_mid = float(np.interp(z_mid, stem[:, 1], stem[:, 0]))
        h0 = min(5.2 * k, 0.8 * (stem[1, 1] - stem[0, 1]))
        plan["name"] = (engrave.fit_name(s.name.strip(), s.font, h0, max_width=2.0 * r_mid), stem, z_mid)
    kind = engrave.FONTS[s.font][1]
    rb = (BASE_R - 0.6) * k - 2.0                                    # flat bottom minus a margin
    lines = [(t, engrave.FONTS[s.font], (0.29 if kind == "script" else 0.22) * rb, engrave.MIN_MSG_MM[kind])
             for t in (s.message1.strip(), s.message2.strip()) if t]
    if s.date.strip():
        lines.append((s.date.strip(), engrave.DATE_FONT, 0.13 * rb, engrave.MIN_DATE_MM))
    if lines:
        plan["under"] = engrave.fit_underside(lines, rb)
    return plan


def check_text(s: RoyalSettings) -> None:
    """Call before a long build: raises ValueError if the text will not fit."""
    _plan_text(s)


def engrave_text(piece, s: RoyalSettings, log: list[str]):
    """Cut the name into the front of the stem and the message/date into the base (piece at
    final size)."""
    plan = _plan_text(s)
    if not plan:
        return piece
    m = _to_manifold(piece)
    if "name" in plan:
        p, stem, z_mid = plan["name"]
        m = m - engrave.stem_tool(p, lambda z: np.interp(z, stem[:, 1], stem[:, 0]), z_mid)
        log.append(f'Name "{p.text}" on the stand: {p.width:.1f} x {p.height:.1f} mm, '
                   f"{engrave.NAME_DEPTH} mm deep")
    if "under" in plan:
        m = m - engrave.underside_tool(plan["under"])
        log.append("Under the base: " + " / ".join(f'"{p.text}" ({p.height:.1f} mm)' for p, _ in plan["under"])
                   + f", {engrave.UNDER_DEPTH} mm deep")
    return _from_manifold(m)


def print_notes(s: RoyalSettings, height: float) -> str:
    lines = [f"ROYAL CHESS - {PIECES[s.piece][0]}  ({s.style}, {height:.0f} mm tall)",
             "", "PRINT", "- Resin (SLA/MSLA), 0.05 mm layers. Print the STL as it is - all text is "
             "already cut into the model.",
             "- Hollow with ~2 mm walls. Put the drain holes on the BACK of the stand or near the "
             "edge of the base - NOT on the text under the base.",
             "- Supports: only on the outer rim of the base and the back - none across the "
             "text under the base or across the face."]
    if s.name.strip() or s.message1.strip() or s.message2.strip() or s.date.strip():
        lines += ["", "ENGRAVED TEXT (check after printing)"]
        if s.name.strip():
            lines.append(f'- Front of the stand: "{s.name.strip()}"  ({s.font})')
        under = [t.strip() for t in (s.message1, s.message2, s.date) if t.strip()]
        if under:
            lines.append("- Under the base (reads when the piece is turned over): "
                         + " / ".join(f'"{t}"' for t in under))
    lines += ["", f"FINISH: {s.finish}",
              "- Metallic finishes: dark base coat into the letters and grooves, metallic paint "
              "on top, lightly rubbed back so the letters and face details stay dark.",
              "- Do not glue a full felt pad over text under the base (use a felt ring)."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- assemble
def _to_manifold(t):
    import manifold3d as mf

    t.merge_vertices()
    m = mf.Manifold(mf.Mesh(vert_properties=np.asarray(t.vertices, np.float32),
                            tri_verts=np.asarray(t.faces, np.uint32)))
    if m.status() != mf.Error.NoError or m.num_tri() == 0:
        raise ValueError(f"mesh not manifold ({m.status()})")
    return m


def _from_manifold(m):
    import trimesh

    o = m.to_mesh()
    return trimesh.Trimesh(np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts), process=False)


def simplify(t, max_faces: int, tol0: float):
    """Watertight-preserving simplification (manifold3d)."""
    try:
        m = _to_manifold(t)
        tol = tol0
        while m.num_tri() > max_faces and tol < 2.0:
            m = m.simplify(tol)
            tol *= 2
        return _from_manifold(m)
    except Exception:
        return t


def build_piece(head_path, s: RoyalSettings, log: list[str], progress=None):
    """Returns (trimesh piece at final size, info dict)."""
    import trimesh

    t0 = time.time()
    check_text(s)                             # fail fast if a name / message is too long
    head_path = Path(head_path)
    m = load_head(head_path)
    log.append(f"Head model: {len(m.faces):,} triangles")
    v = orient(m, head_path.suffix, s.turn, log)
    lm = landmarks(v, s.neck)
    log.append(f"Neck cut at {lm['neck_frac']:.2f} of the model height")
    k = HEAD_H / max(lm["top"] - lm["z_cut"], 1e-9)
    v = np.column_stack([(v[:, 0] - lm["cx"]) * k, (v[:, 1] - lm["cy"]) * k,
                         Z_NECK + (v[:, 2] - lm["z_cut"]) * k])
    pitch = QUALITY_MM.get(s.quality, 0.25) / s.scale
    if progress:
        progress(0.4, desc="Making the head solid...")
    lo = np.array([v[:, 0].min() - 4, v[:, 1].min() - 4, Z_NECK - 1])
    hi = np.array([v[:, 0].max() + 4, v[:, 1].max() + 4, Z_NECK + HEAD_H + 16])
    solid, g = solid_head(v, np.asarray(m.faces), pitch, lo, hi, Z_NECK, log)
    if s.tidy_hair:
        solid = tidy_hair(solid, g, log)
    if progress:
        progress(0.5, desc="Smoothing the face...")
    g.f = head_field(solid, g, v, np.asarray(m.faces), log)
    z_band = Z_NECK + float(s.crown) * HEAD_H
    if progress:
        progress(0.6, desc=f"Fitting the {PIECES[s.piece][0].lower()} regalia...")
    smooth = s.style == "smooth statue"
    {"king": add_crown_smooth if smooth else add_crown,
     "queen": add_tiara_smooth if smooth else add_tiara,
     "bishop": add_mitre}[s.piece](g, solid, z_band)
    head = g.to_trimesh(blur=0)
    del g, solid
    if progress:
        progress(0.75, desc="Bust and pedestal...")
    torso = body(s.piece, max(pitch * 2, 0.14), s.style)
    try:
        piece = _from_manifold(_to_manifold(head) + _to_manifold(torso) + stand(s.style))
    except Exception as exc:  # pragma: no cover - fallback: overlapping shells
        log.append(f"Union failed ({exc}); writing overlapping shells (slicer will merge them)")
        piece = trimesh.util.concatenate([head, torso])
    shells = piece.split(only_watertight=False)
    if len(shells) > 1:                       # drop inner voids (negative volume) and crumbs
        big = max(q.volume for q in shells)
        piece = trimesh.util.concatenate([q for q in shells if q.volume > 0.01 * big])
    piece.apply_scale(s.scale)
    out = simplify(piece, 600_000, 0.01)
    if progress:
        progress(0.85, desc="Engraving the text...")
    out = engrave_text(out, s, log)
    try:                                     # remove sliver triangles: print-shop checkers merge
        out = _from_manifold(_to_manifold(out).simplify(0.005))   # close points and flag them
    except Exception:
        pass
    log.append(f"Piece: {len(out.faces):,} triangles, watertight={out.is_watertight}, "
               f"height {out.bounds[1][2]:.1f} mm ({time.time() - t0:.0f}s)")
    return out, dict(neck=lm["neck_frac"])


def _letter_colours(v, f, colour, s: RoyalSettings | None):
    """Darken the inside of the engraved name, as the antique wash does on the real piece."""
    cols = np.tile(np.array(colour, float), (len(f), 1))
    if s is None or not s.name.strip():
        return cols
    _, _, stem = _stand_profile(s.style)
    stem = stem * s.scale
    c = v[f].mean(axis=1)
    r_s = np.interp(c[:, 2], stem[:, 1], stem[:, 0])
    cut = ((np.hypot(c[:, 0], c[:, 1]) < r_s - 0.15) & (c[:, 2] > stem[0, 1]) & (c[:, 2] < stem[1, 1])
           & (c[:, 1] < 0))
    cols[cut] = np.array(colour, float) * 0.35
    return cols


def render_image(t, colour, path: Path, size_scale: float = 1.0, s: RoyalSettings | None = None) -> Path:
    """Preview picture. The camera depends only on the Size setting (not on the piece), so
    King, Queen and Bishop pictures are at the same scale and can be compared directly."""
    small = simplify(t.copy(), 250_000, 0.02)
    v, f = np.asarray(small.vertices), np.asarray(small.faces)
    hgt = 82.0 * size_scale                     # King incl. cross at this size
    base = mesh_mod.box((hgt * 0.75, hgt * 0.75, hgt * 0.08), (-hgt * 0.375, -hgt * 0.375, -hgt * 0.08))
    items = [render3d.Item(base.vertices, base.faces, (35, 32, 30), 0.4, layer=0),
             render3d.Item(v, f, colour, 0.45, smooth=True, face_colors=_letter_colours(v, f, colour, s))]
    img = render3d.render(items, eye=(hgt * 0.7, -hgt * 2.0, hgt * 1.0), target=(0, 0, hgt * 0.52),
                          size=(800, 1000), fov_deg=34)
    img.save(path)
    return path


def render_underside(t, colour, path: Path) -> Path:
    """The base seen from below, as when the piece is turned over (front edge at the top)."""
    small = simplify(t.copy(), 250_000, 0.02)
    v, f = np.asarray(small.vertices), np.asarray(small.faces)
    rb = float(np.abs(v[:, :2]).max())
    c = v[f].mean(axis=1)
    cols = np.tile(np.array(colour, float), (len(f), 1))
    cols[(c[:, 2] > 0.05) & (c[:, 2] < 1.0) & (np.hypot(c[:, 0], c[:, 1]) < rb * 0.9)] *= 0.35
    flipped = v * [1, -1, -1]                       # turned over, so the lights fall on the base
    img = render3d.render([render3d.Item(flipped, f, colour, 0.3, face_colors=cols)],
                          eye=(0, 0, rb * 4.2), target=(0, 0, 0), size=(700, 700), fov_deg=30,
                          up=(0, 1, 0), background=((60, 52, 44), (40, 34, 28)))
    img.save(path)
    return path


def generate(source, s: RoyalSettings, customer: dict | None = None, api_key: str | None = None,
             progress=None) -> Result:
    """source: path of a head model (GLB/OBJ/STL) or of a photo (needs api_key -> Tripo)."""
    from .providers import tripo

    log: list[str] = []
    customer = customer or {}
    order_id, folder = orders.new_order_dir("royal", customer.get("name", ""))
    source = Path(source)
    if source.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
        head_path = tripo.photo_to_head_model(source, api_key or "", folder, log, progress=progress)
    else:
        head_path = folder / f"head_model{source.suffix.lower()}"
        head_path.write_bytes(source.read_bytes())
    piece, info = build_piece(head_path, s, log, progress)
    label = PIECES[s.piece][0].split()[0].lower()
    stl = folder / f"{order_id}_{label}.stl"
    piece.export(stl)
    colour = FINISH_COLOURS.get(s.finish, (200, 170, 110))
    small = simplify(piece.copy(), 200_000, 0.03)
    glb = mesh_mod.write_glb_preview(mesh_mod.Mesh(np.asarray(small.vertices), np.asarray(small.faces)),
                                     folder / f"{order_id}_{label}_preview.glb", color=colour)
    if progress:
        progress(0.95, desc="Rendering preview...")
    img = render_image(piece, colour, folder / f"{order_id}_{label}.png", SIZES.get(s.size, 1.0), s)
    extra = {}
    if s.message1.strip() or s.message2.strip() or s.date.strip():
        extra["underside"] = render_underside(piece, colour, folder / f"{order_id}_{label}_underside.png")
    notes = folder / f"{order_id}_{label}_print_notes.txt"
    notes.write_text(print_notes(s, float(piece.bounds[1][2])), encoding="utf-8")
    extra["print_notes"] = notes
    vol = float(abs(piece.volume)) if piece.is_watertight else 0.0
    hgt = float(piece.bounds[1][2])
    q = costing.quote_royal(vol, hgt, s.finish, s.packaging)
    p = load_pricing()
    orders.save_order(folder, {"order_id": order_id, "product": f"royal chess {label}",
                               "customer": customer, "quantity": 1,
                               "settings": s.__dict__, "volume_mm3": vol, "height_mm": hgt,
                               "quote": {"total_price": q.total_price, "lines": q.lines},
                               "log": log, **info})
    return Result(order_id, folder, stl, glb, img, q.as_markdown(p["gst_percent"]), log, vol, hgt,
                  head_model=head_path, extra=extra)
