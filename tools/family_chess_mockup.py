"""Family chess set: printable pieces + mockup images from one family photo.

Two styles (--style):
    photo  (default) each piece gets a shallow oval window; a colour photo sheet
           (photo_sheet.png) is produced to print on glossy sticker paper, cut
           and stick in, then cover with a clear epoxy dome.
    relief the face is sculpted in relief (single colour, no stickers).

Faces in the photo are used left to right:
    1st face -> King, 2nd -> Queen, then Bishops, Knights, Rooks (cycling).
Pawns carry initials (--pawns "A,B,C,D,E,F,G,H").

Example:
    python tools/family_chess_mockup.py family.jpg --pawns "A,R,M,S,K,V,P,D"

Output folder (default orders/chess_<timestamp>/):
    king.stl, queen.stl, bishop_1.stl ... pawn_A.stl   (print each once per side)
    photo_sheet.png (photo style: print at 100% on A4 glossy sticker paper)
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


def build_set(faces, pawn_initials, style="photo"):
    """Return {name: (kind, mesh, photo_or_None)} for one side (16 pieces, 11 files)."""
    def face(i):
        return faces[i % len(faces)] if faces else None

    out = {}

    def add(name, kind, f, initial=""):
        photo = None
        if style == "photo" and f is not None:
            photo = chess.window_photo(f, *chess.window_size_mm(kind))
        out[name] = (kind, chess.piece(kind, f, initial, style=style), photo)

    add("king", "king", face(0))
    add("queen", "queen", face(1))
    n = 2
    for kind in ["bishop", "knight", "rook"]:
        for j in (1, 2):
            add(f"{kind}_{j}", kind, face(n))
            n += 1
    for ini in pawn_initials[:8]:
        add(f"pawn_{ini}", "pawn", None, ini)
    return out


def item(entry, offset, colour, shine=0.35, turned_=False):
    kind, m, photo = entry
    fc = chess.photo_face_colors(m, kind, photo, colour) if photo is not None else None
    if turned_:
        m = turned(m)
    return render3d.Item(m.vertices + offset, m.faces, colour, shine, face_colors=fc)


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
    ap.add_argument("--style", choices=["photo", "relief"], default="photo")
    a = ap.parse_args()

    out = Path(a.out) if a.out else ORDERS_DIR / f"chess_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    log = []
    faces = chess.faces_from_photo(a.photo, log) if a.photo else []
    print("\n".join(log))
    initials = [s.strip() for s in a.pawns.split(",") if s.strip()] or list("ABCDEFGH")
    pieces = build_set(faces, (initials * 8)[:8], a.style)

    for name, (kind, m, _) in pieces.items():
        mesh.write_stl(m, out / f"{name}.stl")
    print(f"STL files: {out}")
    photos = [(name.replace("_", " ").title(), ph) for name, (_, _, ph) in pieces.items()
              if ph is not None]
    if photos:
        chess.photo_sheet(photos).save(out / "photo_sheet.png", dpi=(300, 300))
        print("Photo sheet: photo_sheet.png (print at 100% on A4 glossy sticker paper)")

    # 1) the six piece types in a row
    row, x = [], -110
    for kind in chess.PIECES:
        name = next(n for n, (k, _, _) in pieces.items() if k == kind)
        row.append(item(pieces[name], [x, 0, 0], GOLD))
        x += 44
    render3d.render(row, eye=(0, -260, 110), target=(0, 0, 40), size=(1400, 800),
                    fov_deg=38).save(out / "mockup_pieces.png")

    # 2) close-up of king and queen
    render3d.render([item(pieces["king"], [-20, 0, 0], GOLD, 0.4),
                     item(pieces["queen"], [20, 0, 0], GOLD, 0.4)],
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
            entry = pieces[rank_names[kind][used[kind]]]
            used[kind] = min(used[kind] + 1, len(rank_names[kind]) - 1)
            items.append(item(entry, [(c - 3.5) * sq, (back_row - 3.5) * sq, 0], colour,
                              turned_=rot))
        for c in range(8):
            entry = pieces[pawn_names[c % len(pawn_names)]]
            items.append(item(entry, [(c - 3.5) * sq, (pawn_row - 3.5) * sq, 0], colour,
                              turned_=rot))
    render3d.render(items, eye=(0, -520, 330), target=(0, 10, 10), size=(1500, 1000),
                    fov_deg=40).save(out / "mockup_board.png")
    print(f"Mockups: {out}")


if __name__ == "__main__":
    main()
