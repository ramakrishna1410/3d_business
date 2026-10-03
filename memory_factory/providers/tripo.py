"""Tripo3D API (v3): customer photo -> 3D head model, fully automatic.

Flow (from Tripo's v3 docs, https://openapi.tripo3d.ai/v3):
    1. POST /files                       upload the photo  -> file_token
    2. POST /generation/image-to-model   start the job     -> task_id
    3. GET  /tasks/{task_id}             poll every 2 s until status == "success"
    4. download output.model_url (GLB) immediately - the link expires after 5 minutes

The settings below ask for a *bare, detailed geometry* (no colour texture): for a
single-colour print every detail (beard, eyes, hair) has to be in the shape.

The API key is read from the TRIPO_API_KEY environment variable (or a local,
git-ignored .env file) or typed into the Settings tab - never put it in code.
API credits are bought separately from the Tripo app subscription. Uploading a
customer's photo to Tripo needs the customer's consent.
"""

from __future__ import annotations

import time
from pathlib import Path

API_BASE = "https://openapi.tripo3d.ai/v3"
MODEL_VERSION = "v3.1-20260211"

# Pure, detailed geometry for printing. NOTE: pbr defaults to true and forces
# texture back on, so it must be switched off explicitly.
GENERATION_PARAMS = {
    "model": MODEL_VERSION,
    "geometry_quality": "detailed",   # "Ultra" mode - finer geometry
    "texture": False,
    "pbr": False,
    "face_limit": 1_000_000,
    "export_uv": False,
}

DONE_FAIL = ("failed", "cancelled", "canceled", "banned", "expired", "unknown")


class TripoError(RuntimeError):
    pass


def _check(resp, what: str) -> dict:
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if resp.status_code >= 400 or body.get("code", 0) != 0:
        msg = body.get("message") or body.get("msg") or resp.text[:300]
        if resp.status_code in (401, 403):
            msg = f"API key rejected ({resp.status_code}). Check the key in Settings. {msg}"
        raise TripoError(f"Tripo {what} failed (HTTP {resp.status_code}, code {body.get('code')}): {msg}")
    return body.get("data") or {}


def upload_image(path, api_key: str, session=None) -> str:
    import requests

    s = session or requests
    path = Path(path)
    if path.stat().st_size > 20 * 1024 * 1024:
        raise TripoError("Photo is larger than 20 MB - resize it first.")
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    with open(path, "rb") as fh:
        r = s.post(f"{API_BASE}/files", headers={"Authorization": f"Bearer {api_key}"},
                   files={"file": (path.name, fh, mime)}, timeout=120)
    token = _check(r, "upload").get("file_token")
    if not token:
        raise TripoError("Tripo upload returned no file_token.")
    return token


def start_image_to_model(file_token: str, api_key: str, extra: dict | None = None, session=None) -> str:
    import requests

    s = session or requests
    body = {"input": file_token, **GENERATION_PARAMS, **(extra or {})}
    r = s.post(f"{API_BASE}/generation/image-to-model",
               headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
               json=body, timeout=60)
    task_id = _check(r, "image-to-model").get("task_id")
    if not task_id:
        raise TripoError("Tripo returned no task_id.")
    return task_id


def wait_for_task(task_id: str, api_key: str, progress=None, timeout_s: int = 900,
                  poll_s: float = 2.0, session=None, sleep=time.sleep) -> dict:
    import requests

    s = session or requests
    start = time.time()
    while True:
        r = s.get(f"{API_BASE}/tasks/{task_id}", headers={"Authorization": f"Bearer {api_key}"},
                  timeout=60)
        data = _check(r, "task query")
        status = str(data.get("status", "")).lower()
        if progress:
            progress(min(0.9, 0.1 + 0.8 * float(data.get("progress") or 0) / 100),
                     desc=f"Tripo: {status or 'waiting'} {data.get('progress') or 0}%")
        if status == "success":
            return data
        if status in DONE_FAIL:
            raise TripoError(f"Tripo task {status}: {data.get('error') or data.get('message') or ''}")
        if time.time() - start > timeout_s:
            raise TripoError("Tripo took too long (15 min) - try again later.")
        sleep(poll_s)


def download_model(task: dict, out_dir: Path, session=None) -> Path:
    import requests

    s = session or requests
    out = task.get("output") or {}
    url = out.get("model_url") or out.get("pbr_model") or out.get("model")
    if not url and isinstance(out.get("model_urls"), dict):
        url = next(iter(out["model_urls"].values()), None)
    if not url:
        raise TripoError(f"Tripo task finished but returned no model URL ({list(out)}).")
    r = s.get(url, timeout=300)
    if r.status_code >= 400:
        raise TripoError(f"Model download failed (HTTP {r.status_code}) - the link may have expired.")
    path = Path(out_dir) / "tripo_head.glb"
    path.write_bytes(r.content)
    return path


def prepare_photo(photo_path, out_dir: Path, max_side: int = 2048) -> Path:
    """Upright, RGB, at most 2048 px, saved as JPEG (Tripo accepts JPEG/PNG up to 20 MB)."""
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
    img.thumbnail((max_side, max_side))
    out = Path(out_dir) / "photo_for_tripo.jpg"
    img.save(out, quality=95)
    return out


def photo_to_head_model(photo_path, api_key: str, out_dir: Path, log: list[str],
                        progress=None, session=None, sleep=time.sleep) -> Path:
    """Upload a photo, generate the 3D head, download the GLB. Returns its path."""
    if not api_key:
        raise TripoError("No Tripo API key. Add it in Settings (or set TRIPO_API_KEY).")
    t0 = time.time()
    if progress:
        progress(0.03, desc="Uploading photo to Tripo...")
    token = upload_image(prepare_photo(photo_path, out_dir), api_key, session=session)
    task_id = start_image_to_model(token, api_key, session=session)
    log.append(f"Tripo task {task_id} started ({MODEL_VERSION}, Ultra geometry, no texture)")
    task = wait_for_task(task_id, api_key, progress=progress, session=session, sleep=sleep)
    path = download_model(task, out_dir, session=session)
    credits = task.get("credits_consumed")
    log.append(f"Tripo model downloaded in {time.time() - t0:.0f}s"
               + (f", {credits} credits used" if credits is not None else ""))
    return path
