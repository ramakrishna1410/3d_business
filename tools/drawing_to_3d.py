"""Command-line kids' drawing -> 3D (same engine as the app).

Examples:
    python tools/drawing_to_3d.py dino.jpg --name Aarav --line2 "Age 6 - 2026"
    python tools/drawing_to_3d.py dino.jpg --mode "standing figure" --size 120
    python tools/drawing_to_3d.py dino.jpg --mode "import 3d model" --model tripo.glb
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_factory import drawing  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--name", default="")
    ap.add_argument("--line2", default="")
    ap.add_argument("--mode", choices=drawing.MODES, default="relief plaque")
    ap.add_argument("--size", type=float, default=100, help="longest side / height in mm")
    ap.add_argument("--puff", type=float, default=6)
    ap.add_argument("--model", help="GLB/OBJ/STL for 'import 3d model'")
    ap.add_argument("--paint", default="hand painted")
    ap.add_argument("--packaging", default="display box")
    ap.add_argument("--customer", default="")
    ap.add_argument("--phone", default="")
    a = ap.parse_args()

    s = drawing.DrawingSettings(child_name=a.name, age_line=a.line2, mode=a.mode,
                                size_mm=a.size, puff_mm=a.puff)
    r = drawing.generate(a.image, s, {"name": a.customer, "phone": a.phone, "consent": True},
                         a.paint, a.packaging, model_file=a.model)
    print("\n".join(r.log))
    print(f"\nSTL:     {r.stl}\nPreview: {r.render}\n")
    print(r.quote_md)


if __name__ == "__main__":
    main()
