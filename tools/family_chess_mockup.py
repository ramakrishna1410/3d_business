"""Family chess set: printable pieces + mockup images from one family photo.

Faces in the photo are used left to right:
    1st face -> King, 2nd -> Queen, then Bishops, Knights, Rooks (cycling).
Pawns carry initials (--pawns "A,B,C,D,E,F,G,H").

Example:
    python tools/family_chess_mockup.py family.jpg --pawns "A,R,M,S,K,V,P,D"

Output folder (default orders/chess_<timestamp>/):
    king.stl, queen.stl, bishop_1.stl ... pawn_A.stl   (print each once per side)
    mockup_pieces.png, mockup_closeup.png, mockup_board.png
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from memory_factory import chess, mesh, render3d  # noqa: E402
from memory_factory.config import ORDERS_DIR  # noqa: E402

GOLD = (214, 168, 82)
SILVER = (170, 172, 178)


def build_set(faces, pawn_initials):
    """Return {name: (kind, mesh)} for one side (16 pieces, 11 unique files)."""
    def face(i):
        return faces[i % len(faces)] if faces else None

    out = {"king": ("king", chess.piece("king", face(0))),
           "queen": ("queen", chess.piece("queen", face(1)))}
    order = ["bishop", "knight", "rook"]
    n = 2
    for kind in order:
        for j in (1, 2):
            out[f"{kind}_{j}"] = (kind, chess.piece(kind, face(n)))
            n += 1
    for ini in pawn_initials[:8]:
        out[f"pawn_{ini}"] = ("pawn", chess.piece("pawn", None, ini))
    return out


def board_items(sq):
    items = []
    frame = mesh.box((8 * sq + 24, 8 * sq + 24, 8), (-4 * sq - 12, -4 * sq - 12, -14))
    items.append(render3d.Item(frame.vertices, frame.faces, (70, 45, 28), 0.05, layer=-1))
    for r in range(8):
        for c in range(8):
            col = (222, 200, 160) if (r + c) % 2 else (120, 82, 52)
            b = mesh.box((sq, sq, 6), ((c - 4) * sq, (r - 4) * sq, -6))
            items.append(render3d.Item(b.vertices, b.faces, col, 0.05, layer=0))
    return items


def turned(m):
    v = m.vertices.copy()
    v[:, :2] *= -1  # rotate 180 deg so black's cameos face the black player
    return mesh.Mesh(v, m.faces)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo", nargs="?", help="family photo (faces left to right)")
    ap.add_argument("--pawns", default="A,B,C,D,E,F,G,H", help="8 initials for the pawns")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    out = Path(a.out) if a.out else ORDERS_DIR / f"chess_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    log = []
    faces = chess.faces_from_photo(a.photo, log) if a.photo else []
    print("\n".join(log))
    initials = [s.strip() for s in a.pawns.split(",") if s.strip()] or list("ABCDEFGH")
    pieces = build_set(faces, (initials * 8)[:8])

    for name, (kind, m) in pieces.items():
        mesh.write_stl(m, out / f"{name}.stl")
    print(f"STL files: {out}")

    # 1) the six piece types in a row
    row, x = [], -110
    for kind in chess.PIECES:
        name = next(n for n, (k, _) in pieces.items() if k == kind)
        m = pieces[name][1]
        row.append(render3d.Item(m.vertices + [x, 0, 0], m.faces, GOLD, 0.35))
        x += 44
    render3d.render(row, eye=(0, -260, 110), target=(0, 0, 40), size=(1400, 800),
                    fov_deg=38).save(out / "mockup_pieces.png")

    # 2) close-up of king and queen
    k, q = pieces["king"][1], pieces["queen"][1]
    render3d.render([render3d.Item(k.vertices + [-20, 0, 0], k.faces, GOLD, 0.4),
                     render3d.Item(q.vertices + [20, 0, 0], q.faces, GOLD, 0.4)],
                    eye=(0, -150, 55), target=(0, 0, 42), size=(1200, 900),
                    fov_deg=34).save(out / "mockup_closeup.png")

    # 3) full board: gold vs silver
    sq = chess.SQUARE_MM
    items = board_items(sq)
    rank_names = {"rook": ["rook_1", "rook_2"], "knight": ["knight_1", "knight_2"],
                  "bishop": ["bishop_1", "bishop_2"], "queen": ["queen"], "king": ["king"]}
    pawn_names = [n for n in pieces if n.startswith("pawn_")]
    for colour, back_row, pawn_row, rot in [(GOLD, 0, 1, False), (SILVER, 7, 6, True)]:
        used = {k: 0 for k in rank_names}
        for c, kind in enumerate(chess.BACK_RANK):
            m = pieces[rank_names[kind][used[kind]]][1]
            used[kind] = min(used[kind] + 1, len(rank_names[kind]) - 1)
            m = turned(m) if rot else m
            items.append(render3d.Item(m.vertices + [(c - 3.5) * sq, (back_row - 3.5) * sq, 0],
                                       m.faces, colour, 0.35))
        for c in range(8):
            m = pieces[pawn_names[c % len(pawn_names)]][1]
            m = turned(m) if rot else m
            items.append(render3d.Item(m.vertices + [(c - 3.5) * sq, (pawn_row - 3.5) * sq, 0],
                                       m.faces, colour, 0.35))
    render3d.render(items, eye=(0, -520, 330), target=(0, 10, 10), size=(1500, 1000),
                    fov_deg=40).save(out / "mockup_board.png")
    print(f"Mockups: {out}")


if __name__ == "__main__":
    main()
