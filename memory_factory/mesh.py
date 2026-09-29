"""Turn height maps into watertight, printable triangle meshes.

Every product in this project ends up as a 2D height map (a numpy array in
millimetres).  This module converts such a map into a closed solid:

    top surface  = the height map
    bottom       = flat (plaques, medallions) or a mirrored height map
                   (free-standing sculptures)
    side walls   = generated along the outline of the mask

The result is written as a binary STL, which every resin/FDM slicer
(Chitubox, Lychee, PrusaSlicer, Cura...) opens directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray  # (N, 3) float, millimetres
    faces: np.ndarray  # (M, 3) int, counter-clockwise seen from outside

    @property
    def triangle_count(self) -> int:
        return int(len(self.faces))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def volume_mm3(self) -> float:
        """Signed volume via the divergence theorem (exact for closed meshes)."""
        v = self.vertices[self.faces]
        return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)

    def translated(self, offset) -> "Mesh":
        return Mesh(self.vertices + np.asarray(offset, dtype=float), self.faces.copy())

    def rotated_x(self, degrees: float) -> "Mesh":
        a = np.radians(degrees)
        rot = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        return Mesh(self.vertices @ rot.T, self.faces.copy())


def combine(meshes: list[Mesh]) -> Mesh:
    """Put several closed shells into one file (slicers merge overlapping shells)."""
    verts, faces, offset = [], [], 0
    for m in meshes:
        verts.append(m.vertices)
        faces.append(m.faces + offset)
        offset += len(m.vertices)
    return Mesh(np.vstack(verts), np.vstack(faces))


def _remove_pinches(cell: np.ndarray) -> np.ndarray:
    """Drop cells that touch their neighbours only at a corner.

    Two cells meeting diagonally share a single vertex, which makes the solid
    non-manifold (slicers complain). Removing one of them fixes it.
    """
    cell = cell.copy()
    for _ in range(100):
        a = cell[:-1, :-1] & cell[1:, 1:] & ~cell[:-1, 1:] & ~cell[1:, :-1]
        b = cell[:-1, 1:] & cell[1:, :-1] & ~cell[:-1, :-1] & ~cell[1:, 1:]
        if not (a.any() or b.any()):
            break
        cell[:-1, :-1][a] = False
        cell[:-1, 1:][b] = False
    return cell


def heightmap_to_mesh(
    top: np.ndarray,
    mask: np.ndarray | None = None,
    pixel_mm: float = 0.15,
    bottom: np.ndarray | float = 0.0,
) -> Mesh:
    """Build a closed solid from a height map.

    top     : (H, W) heights in mm of the upper surface.
    mask    : (H, W) bool, vertices that belong to the object. A grid cell is
              kept only when all four of its corners are inside the mask.
    pixel_mm: size of one pixel in mm.
    bottom  : scalar z of a flat underside, or an (H, W) array (e.g. -top for a
              double-sided sculpture). Must be below `top` everywhere in the mask.
    """
    top = np.asarray(top, dtype=np.float64)
    h, w = top.shape
    if mask is None:
        mask = np.ones((h, w), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    bot = np.full((h, w), float(bottom)) if np.isscalar(bottom) else np.asarray(bottom, float)

    cell = _remove_pinches(mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:])
    if not cell.any():
        raise ValueError("Mask is empty - nothing to build.")

    # Only keep vertices that are used by at least one cell.
    used = np.zeros((h, w), dtype=bool)
    used[:-1, :-1] |= cell
    used[:-1, 1:] |= cell
    used[1:, :-1] |= cell
    used[1:, 1:] |= cell
    idx = -np.ones((h, w), dtype=np.int64)
    n = int(used.sum())
    idx[used] = np.arange(n)

    rows, cols = np.nonzero(used)
    x = cols * pixel_mm
    y = (h - 1 - rows) * pixel_mm  # image top -> +y
    v_top = np.column_stack([x, y, top[rows, cols]])
    v_bot = np.column_stack([x, y, bot[rows, cols]])
    vertices = np.vstack([v_top, v_bot])

    ci, cj = np.nonzero(cell)
    v00 = idx[ci, cj]
    v01 = idx[ci, cj + 1]
    v10 = idx[ci + 1, cj]
    v11 = idx[ci + 1, cj + 1]
    # Counter-clockwise when viewed from +z (image-down is -y).
    top_faces = np.vstack(
        [np.column_stack([v10, v11, v01]), np.column_stack([v10, v01, v00])]
    )
    bot_faces = top_faces[:, ::-1] + n

    # Boundary = directed quad edges whose reverse does not exist.
    e = np.vstack(
        [
            np.column_stack([v10, v11]),
            np.column_stack([v11, v01]),
            np.column_stack([v01, v00]),
            np.column_stack([v00, v10]),
        ]
    )
    fwd = e[:, 0] * (n + 1) + e[:, 1]
    rev = e[:, 1] * (n + 1) + e[:, 0]
    boundary = e[~np.isin(fwd, rev)]
    a, b = boundary[:, 0], boundary[:, 1]
    wall = np.vstack([np.column_stack([a, b + n, b]), np.column_stack([a, a + n, b + n])])

    faces = np.vstack([top_faces, bot_faces, wall]).astype(np.int64)
    return Mesh(vertices, faces)


def box(size, origin=(0.0, 0.0, 0.0)) -> Mesh:
    sx, sy, sz = size
    ox, oy, oz = origin
    v = np.array(
        [[0, 0, 0], [sx, 0, 0], [sx, sy, 0], [0, sy, 0],
         [0, 0, sz], [sx, 0, sz], [sx, sy, sz], [0, sy, sz]],
        dtype=float,
    ) + [ox, oy, oz]
    f = np.array(
        [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
         [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    )
    return Mesh(v, f)


def write_stl(mesh: Mesh, path: str | Path, name: str = "memory_factory") -> Path:
    path = Path(path)
    tri = mesh.vertices[mesh.faces].astype(np.float32)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lens, out=np.zeros_like(normals), where=lens > 0)

    record = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    data = np.zeros(len(tri), dtype=record)
    data["n"] = normals
    data["v"] = tri
    header = name.encode("ascii", "ignore")[:80].ljust(80, b" ")
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(np.uint32(len(tri)).tobytes())
        fh.write(data.tobytes())
    return path


def write_glb_preview(mesh: Mesh, path: str | Path, color=(200, 170, 110),
                      texture=None, extent_mm=None, upright: bool = False) -> Path | None:
    """Lightweight GLB for the in-browser 3D viewer (needs `trimesh`).

    texture  : optional PIL image (e.g. the shaded render) painted onto the
               model as vertex colours, so fine relief detail is visible even
               under the viewer's flat lighting.
    extent_mm: (width, height) in mm that the texture image covers; the
               texture is mapped over the model's x/y from 0 to that size.
    upright  : True for plaques/coins: stand the piece up facing the viewer
               instead of lying flat on the floor.
    """
    try:
        import trimesh
    except ImportError:
        return None
    tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    if texture is not None:
        tex = np.asarray(texture.convert("RGB"))
        th, tw = tex.shape[:2]
        ex, ey = extent_mm or (mesh.vertices[:, 0].max(), mesh.vertices[:, 1].max())
        col = np.clip(mesh.vertices[:, 0] / max(ex, 1e-9) * (tw - 1), 0, tw - 1).astype(int)
        row = np.clip((1 - mesh.vertices[:, 1] / max(ey, 1e-9)) * (th - 1), 0, th - 1).astype(int)
        rgb = tex[row, col]
        tm.visual.vertex_colors = np.column_stack([rgb, np.full(len(rgb), 255)]).astype(np.uint8)
    else:
        tm.visual.face_colors = [*color, 255]
    if not upright:
        # Viewer is y-up; our standing models are z-up.
        tm.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    tm.apply_translation(-tm.bounds.mean(axis=0))
    tm.export(path)
    return Path(path)


def is_watertight(mesh: Mesh) -> bool:
    """Every edge must be shared by exactly two faces with opposite direction."""
    f = mesh.faces
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    n = len(mesh.vertices) + 1
    fwd = e[:, 0] * n + e[:, 1]
    rev = e[:, 1] * n + e[:, 0]
    uniq, counts = np.unique(fwd, return_counts=True)
    return bool((counts == 1).all() and np.isin(fwd, rev).all())
