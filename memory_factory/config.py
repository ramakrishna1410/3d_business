"""Paths and editable business settings (prices, material costs)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
ORDERS_DIR = ROOT / "orders"
PRICING_FILE = CONFIG_DIR / "pricing.json"


def load_pricing() -> dict:
    with open(PRICING_FILE, encoding="utf-8") as fh:
        return json.load(fh)
