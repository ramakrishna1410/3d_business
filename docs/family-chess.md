# Family Chess Set (concept & prototype)

A full chess set in which every piece carries a **relief portrait (cameo) of a family member**.
You get classic, instantly recognisable pieces, personalised with the family's faces.

| Piece | Suggested person |
|---|---|
| King / Queen | Grandparents or the couple |
| Bishops | Uncles and aunts |
| Knights | Children |
| Rooks | Elders, or the family home and temple (a photo relief) |
| Pawns | Cousins, grandchildren, pets (face or initial) |

White side = one family, black side = the other: great as a **wedding gift that brings the
two families together**, a 60th-birthday gift, or a grandparents' gift.

## Make it
```
python tools/family_chess_mockup.py family_photo.jpg --pawns "A,R,M,S,K,V,P,D"
```
Faces are taken left to right (1st = King, 2nd = Queen, then Bishops, Knights, Rooks).
Output: one STL per unique piece plus three mockup images (pieces, close-up, full board).

## Printing
- 32 pieces: print each side's 16 pieces in its colour (e.g. gold silk PLA vs silver silk PLA).
- Standing upright, **0.08–0.12 mm layers**, supports only under the queen's crown balls and the
  king's cross arms if the slicer asks. King ≈ 92 mm tall, pawns ≈ 46 mm.
- Faces are small (cameos 14–25 mm), so they look best at 0.08 mm on FDM, or on resin for a
  premium set.
- Board: a wooden board (42–45 mm squares) bought locally, or engraved with the family name.
  A printed board is too big for most printers.

## Rough cost & price (to validate with quotes)
| | ₹ |
|---|---|
| 32 pieces FDM silk PLA (≈ ₹40–120 each at a print service) | ~2,000–3,000 |
| Wooden board + box + felt bases | ~800–1,500 |
| **Cost** | **~3,000–4,500** |
| **Suggested price** | **₹6,999–9,999** (premium resin set ₹14,999+) |

With your own printer, the pieces cost only ~₹300–500 in material, so the margin becomes very large.
