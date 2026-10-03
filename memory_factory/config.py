"""Paths and editable business settings (prices, material costs)."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
ORDERS_DIR = ROOT / "orders"
PRICING_FILE = CONFIG_DIR / "pricing.json"


def load_pricing() -> dict:
    with open(PRICING_FILE, encoding="utf-8") as fh:
        return json.load(fh)


def load_local_env(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from the git-ignored .env file (e.g. TRIPO_API_KEY)."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
