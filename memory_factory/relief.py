"""Depth map -> bas-relief height map.

A raw depth map is a poor relief: the big difference between the subject
and the background eats the whole depth budget (1-2 mm on a medallion), and
the small details that carry likeness - eyes, lips, smile lines - vanish.

We use gradient-domain compression, the technique behind professional
digital bas-relief:

1. take the slopes (gradients) of the depth map
2. delete the huge jumps at silhouette edges
3. squash big slopes, keep small ones (detail) almost unchanged
4. rebuild the surface from the modified slopes (Poisson solve)
5. add a little fine detail from the photo's shading and
   fade the edges into the background plane
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.fft import dctn, idctn


def _poisson_neumann(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Least-squares surface whose forward differences best match (gx, gy)."""
    h, w = gx.shape
    div = np.zeros((h, w))
    div[:, 0] += gx[:, 0]
    div[:, 1:] += gx[:, 1:] - gx[:, :-1]
    div[0, :] += gy[0, :]
    div[1:, :] += gy[1:, :] - gy[:-1, :]
    # Forward difference of the last column/row is defined as 0.
    div[:, -1] -= gx[:, -1]
    div[-1, :] -= gy[-1, :]

    d = dctn(div, type=2, norm="ortho")
    yy = np.cos(np.pi * np.arange(h) / h)[:, None]
    xx = np.cos(np.pi * np.arange(w) / w)[None, :]
    denom = (2 * yy - 2) + (2 * xx - 2)
    denom[0, 0] = 1.0
    u = d / denom
    u[0, 0] = 0.0
    return idctn(u, type=2, norm="ortho")


def bas_relief(
    depth: np.ndarray,
    mask: np.ndarray,
    gray: np.ndarray | None = None,
    compression: float = 0.5,
    detail: float = 1.0,
    edge_softness_px: float | None = None,
) -> np.ndarray:
    """Return relief heights 0..1 (0 = background plane).

    compression: 0 (almost flat, coin-like) .. 1 (close to the raw depth)
    detail     : multiplier for fine texture (hair, eyes, fabric)
    """
    depth = np.asarray(depth, dtype=np.float64)
    soft = np.clip(np.asarray(mask, dtype=np.float64), 0, 1)
    hard = soft > 0.5
    if not hard.any():
        return np.zeros_like(depth)

    gx = np.zeros_like(depth)
    gy = np.zeros_like(depth)
    gx[:, :-1] = depth[:, 1:] - depth[:, :-1]
    gy[:-1, :] = depth[1:, :] - depth[:-1, :]

    # 1-2: kill silhouette cliffs and everything outside the subject.
    inside_x = np.zeros_like(hard)
    inside_y = np.zeros_like(hard)
    inside_x[:, :-1] = hard[:, 1:] & hard[:, :-1]
    inside_y[:-1, :] = hard[1:, :] & hard[:-1, :]
    mag = np.hypot(gx, gy)
    tau = np.percentile(mag[hard], 98) * 3 + 1e-9
    gx = np.where(inside_x & (np.abs(gx) < tau), gx, 0.0)
    gy = np.where(inside_y & (np.abs(gy) < tau), gy, 0.0)

    # 3: attenuate large slopes more than small ones.
    alpha = np.percentile(np.hypot(gx, gy)[hard], 75) * (0.3 + 2.0 * compression) + 1e-9
    mag = np.hypot(gx, gy)
    factor = alpha / (mag + alpha)
    gx *= factor
    gy *= factor

    # 4: rebuild.
    rel = _poisson_neumann(gx, gy)
    rel -= np.percentile(rel[hard], 1)

    # 5: fine detail from the photo and soft blending into the background.
    if gray is not None and detail > 0:
        sigma = max(1.0, min(depth.shape) / 250)
        hp = gray - ndimage.gaussian_filter(gray, sigma * 4)
        scale = np.percentile(rel[hard], 99) + 1e-9
        rel += hp * detail * 0.25 * scale
    base_structure = ndimage.gaussian_filter(rel, min(depth.shape) / 40)
    rel = base_structure + (rel - base_structure) * (1 + 0.5 * detail)

    rel = np.clip(rel / (np.percentile(rel[hard], 99.5) + 1e-9), 0, 1)

    softness = edge_softness_px or max(2.0, min(depth.shape) / 60)
    dist = ndimage.distance_transform_edt(hard)
    ramp = np.clip(dist / softness, 0, 1)
    ramp = ramp * ramp * (3 - 2 * ramp)  # smoothstep
    return rel * ramp


def inflate(mask: np.ndarray, roundness: float = 1.0) -> np.ndarray:
    """Balloon-like dome over a flat shape (0..1). Used for kids' drawings."""
    hard = np.asarray(mask) > 0.5
    dist = ndimage.distance_transform_edt(hard)
    if dist.max() == 0:
        return np.zeros(hard.shape)
    r = dist.max()
    t = np.clip(dist / r, 0, 1)
    dome = np.sqrt(np.clip(t * (2 - t), 0, 1))
    # Mix in a local-thickness term so small parts (arms, tails) stay visible.
    local = np.clip(dist / (np.percentile(dist[hard], 60) + 1e-9), 0, 1)
    local = np.sqrt(np.clip(local * (2 - local), 0, 1))
    out = dome * 0.65 + local * 0.35
    out = ndimage.gaussian_filter(out, max(1.0, r / 25)) * hard
    return np.clip(out ** (1.0 / max(roundness, 0.2)), 0, 1)
