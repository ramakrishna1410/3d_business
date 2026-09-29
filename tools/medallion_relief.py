"""Command-line medallion generator (same engine as the app).

Example:
    python tools/medallion_relief.py couple.jpg --names "Ramesh & Meera" \
        --date 22.04.2026 --qty 150 --hole
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_factory import medallion  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo")
    ap.add_argument("--names", default="Ramesh & Meera")
    ap.add_argument("--date", default="")
    ap.add_argument("--message", default="With Love & Thanks")
    ap.add_argument("--diameter", type=float, default=50)
    ap.add_argument("--relief", type=float, default=1.4, help="face relief height in mm")
    ap.add_argument("--thickness", type=float, default=2.5, help="coin thickness in mm")
    ap.add_argument("--hole", action="store_true", help="add keychain/ribbon hole")
    ap.add_argument("--engine", choices=["auto", "ai", "fast"], default="auto")
    ap.add_argument("--no-bg-removal", action="store_true")
    ap.add_argument("--qty", type=int, default=100)
    ap.add_argument("--finish", default="antique brass paint")
    ap.add_argument("--packaging", default="velvet pouch")
    ap.add_argument("--customer", default="")
    ap.add_argument("--phone", default="")
    a = ap.parse_args()

    s = medallion.MedallionSettings(
        names=a.names, date=a.date, extra=a.message, diameter_mm=a.diameter,
        relief_mm=a.relief, base_mm=a.thickness, keychain_hole=a.hole,
        depth_engine=a.engine, remove_background=not a.no_bg_removal)
    r = medallion.generate(a.photo, s, {"name": a.customer, "phone": a.phone, "consent": True},
                           a.qty, a.finish, a.packaging)
    print("\n".join(r.log))
    print(f"\nSTL:   {r.stl}\nProof: {r.proof}\n")
    print(r.quote_md)


if __name__ == "__main__":
    main()
