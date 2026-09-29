"""Very small order book: one folder per order with all files + order.json."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path

from .config import ORDERS_DIR


def new_order_dir(product: str, customer: str = "") -> tuple[str, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", customer).strip("-")[:30] or "walkin"
    order_id = f"{stamp}-{product[:3].upper()}-{uuid.uuid4().hex[:4]}"
    path = ORDERS_DIR / f"{order_id}_{slug}"
    path.mkdir(parents=True, exist_ok=True)
    return order_id, path


def save_order(path: Path, data: dict) -> Path:
    data = {"saved_at": datetime.now().isoformat(timespec="seconds"), **data}
    out = path / "order.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def list_orders(limit: int = 200) -> list[dict]:
    if not ORDERS_DIR.exists():
        return []
    rows = []
    for f in sorted(ORDERS_DIR.glob("*/order.json"), reverse=True)[:limit]:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(
            {
                "order_id": d.get("order_id", f.parent.name),
                "date": d.get("saved_at", ""),
                "product": d.get("product", ""),
                "customer": d.get("customer", {}).get("name", ""),
                "phone": d.get("customer", {}).get("phone", ""),
                "qty": d.get("quantity", 1),
                "price": d.get("quote", {}).get("total_price", ""),
                "status": d.get("status", "proof sent"),
                "folder": str(f.parent),
            }
        )
    return rows
