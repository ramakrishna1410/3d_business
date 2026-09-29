"""Optional cloud AI: Meshy image-to-3D, plus import of any downloaded model.

Two ways to get a full 3D figure from a drawing:

1. API (automatic): set MESHY_API_KEY (Meshy account -> API settings, paid
   credits). `image_to_printable_mesh` uploads the drawing, waits for the
   model, downloads it and makes it printable.
2. Manual (works with Tripo, Meshy or any tool): generate the model on the
   tool's website, download GLB/OBJ/STL, and load it with
   `model_file_to_printable_mesh`. The app's "Import 3D model" option does this.

API details follow Meshy's public docs (POST /openapi/v1/image-to-3d, poll
GET /openapi/v1/image-to-3d/{id}, result in `model_urls`). Check
https://docs.meshy.ai if the service changes. Uploading customer images to a
cloud service requires the customer's consent.
"""

from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .. import mesh as mesh_mod

API_BASE = "https://api.meshy.ai/openapi/v1/image-to-3d"


def _data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def create_model(img: Image.Image, api_key: str, out_dir: Path, log: list[str],
                 progress=None, timeout_s: int = 900) -> Path:
    import requests

    headers = {"Authorization": f"Bearer {api_key}"}
    r = requests.post(API_BASE, headers=headers, json={"image_url": _data_uri(img)}, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"Meshy rejected the request ({r.status_code}): {r.text[:300]}")
    task_id = r.json().get("result")
    log.append(f"Meshy task {task_id} started")

    start = time.time()
    while True:
        t = requests.get(f"{API_BASE}/{task_id}", headers=headers, timeout=60).json()
        status = t.get("status")
        if progress:
            progress(min(0.95, t.get("progress", 0) / 100), desc=f"Meshy: {status}")
        if status == "SUCCEEDED":
            break
        if status in ("FAILED", "CANCELED", "EXPIRED"):
            raise RuntimeError(f"Meshy task {status}: {t.get('task_error')}")
        if time.time() - start > timeout_s:
            raise TimeoutError("Meshy took too long - try again later.")
        time.sleep(5)

    urls = t.get("model_urls", {})
    url = urls.get("glb") or urls.get("obj")
    if not url:
        raise RuntimeError(f"Meshy returned no model URL: {list(urls)}")
    path = out_dir / ("meshy_model.glb" if urls.get("glb") else "meshy_model.obj")
    path.write_bytes(requests.get(url, timeout=300).content)
    log.append(f"Meshy model downloaded ({time.time() - start:.0f}s)")
    return path


def model_file_to_printable_mesh(model_path, target_height_mm: float,
                                 name_band, pixel_mm: float,
                                 log: list[str], y_up: bool | None = None) -> mesh_mod.Mesh:
    """Load GLB/OBJ/STL/PLY, clean it, scale it, and stand it on a name base.

    name_band: text mask array, or a function width_mm -> text mask array.
    """
    import trimesh

    model_path = Path(model_path)
    loaded = trimesh.load(model_path, force="mesh", process=True)
    if y_up is None:
        y_up = model_path.suffix.lower() in (".glb", ".gltf", ".obj")
    if y_up:  # GLB/OBJ from AI tools are y-up; we print z-up.
        loaded.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))

    loaded.merge_vertices()
    loaded.update_faces(loaded.nondegenerate_faces())
    trimesh.repair.fix_normals(loaded)
    trimesh.repair.fill_holes(loaded)
    parts = loaded.split(only_watertight=False)
    if len(parts) > 1:  # drop floating crumbs the AI sometimes adds
        biggest = max(p.area for p in parts)
        loaded = trimesh.util.concatenate([p for p in parts if p.area > 0.02 * biggest])
    log.append(f"Imported model: {len(loaded.faces):,} faces, watertight={loaded.is_watertight}"
               + ("" if loaded.is_watertight else " (slicer auto-repair recommended)"))

    ext = loaded.bounds[1] - loaded.bounds[0]
    loaded.apply_scale(target_height_mm / max(ext[2], 1e-6))
    lo, hi = loaded.bounds
    fig = mesh_mod.Mesh(np.asarray(loaded.vertices, float), np.asarray(loaded.faces, np.int64))

    width, depth = hi[0] - lo[0], hi[1] - lo[1]
    base_h, sink = 8.0, 1.5
    base = mesh_mod.box((width + 12, depth + 12, base_h))
    fig = fig.translated([-lo[0] + 6, -lo[1] + 6, -lo[2] + base_h - sink])
    parts_out = [base, fig]

    band = name_band(width + 8) if callable(name_band) else name_band
    if band is not None and band.any():
        from scipy import ndimage

        h = ndimage.gaussian_filter(band, 0.7) + 1.0
        slab = mesh_mod.heightmap_to_mesh(np.pad(h, 1, constant_values=1.0),
                                          np.ones((h.shape[0] + 2, h.shape[1] + 2), bool),
                                          pixel_mm, bottom=0.0).rotated_x(90)
        s0, s1 = slab.bounds()
        slab_h = s1[2] - s0[2]
        if slab_h + 1 > base_h:
            base_h2 = slab_h + 1
            parts_out[0] = mesh_mod.box((width + 12, depth + 12, base_h2))
            parts_out[1] = fig.translated([0, 0, base_h2 - base_h])
            base_h = base_h2
        parts_out.append(slab.translated([-s0[0] + 2, -s1[1] + 0.8, -s0[2] + (base_h - slab_h) / 2]))
    return mesh_mod.combine(parts_out)


def image_to_printable_mesh(img: Image.Image, api_key: str | None, target_height_mm: float,
                            name_band, pixel_mm: float, out_dir: Path, log: list[str],
                            progress=None) -> mesh_mod.Mesh:
    api_key = api_key or os.environ.get("MESHY_API_KEY")
    if not api_key:
        raise RuntimeError("AI full 3D needs a Meshy API key (Settings tab or MESHY_API_KEY). "
                           "Or generate the model on meshy.ai / tripo3d.ai and use "
                           "'Import 3D model' instead.")
    path = create_model(img, api_key, out_dir, log, progress)
    return model_file_to_printable_mesh(path, target_height_mm, name_band, pixel_mm, log)
