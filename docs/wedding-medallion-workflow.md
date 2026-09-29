# Wedding Return-Gift Medallion: Workflow (SOP)

**Product:** a coin, 40–60 mm across, with the couple's faces in raised relief, their names,
the date and a short message, finished in antique brass or bronze, in a pouch or box.
**Sold in batches:** 50–500 pieces per wedding.

```
ENQUIRY ─► PHOTO ─► AUTO DESIGN (app) ─► PROOF ON WHATSAPP ─► APPROVAL + ADVANCE
   ─► TEST COIN ─► BATCH PRINT ─► WASH & CURE ─► FINISH ─► QC ─► PACK ─► DELIVER
```

## 1. Enquiry (Day 0)

Collect:
- Names exactly as they should appear (check spelling with the family), the date and the message
- Quantity; event date; delivery address
- Size (default 50 mm), finish, packaging, and whether it needs a keychain/ribbon hole
- **Lead time rule:** at least 10 days before the event for up to 200 pieces, and
  3 weeks for 200–500. Printer hours are shown in the quote.

## 2. The photo

What works best:
- Both faces clearly visible, looking roughly at the camera, heads close together
- Good even light, no strong shadows across the faces, no sunglasses
- A plain background if possible (the app removes busy ones, but plain is always better)
- The original file from the phone, not a WhatsApp-compressed forward

What doesn't work: group photos (faces become too small on a coin), side profiles, and
heavy filters.

## 3. Automatic design (app → 💍 Wedding Medallion)

1. Upload the photo and enter the names, date, message, quantity, finish and packaging.
2. Fill in the customer details and **tick consent**.
3. Click **Generate medallion**. In about 5–30 seconds you get:
   - `proof_for_customer.jpg`: the approval card
   - the STL file: what the printer needs
   - a 3D preview you can rotate
   - the cost breakdown and suggested price
4. Tweak if needed (*Design settings*):
   | Problem | Fix |
   |---|---|
   | Faces look flat | Increase *Face relief height* (up to 2.0 mm) or move *Depth* toward "deep" |
   | Faces look swollen or "balloony" | Move *Depth* toward "flat" (0.3) |
   | Hair and eyes are mushy | Increase *Fine detail* to 1.3–1.5 |
   | Text too thin to print | Increase *Letter height* to 0.9–1.0 mm, and use short names |
   | Background left over | Tick *Remove background*; install the AI engines (README) |

**Quality rule:** faces on a 50 mm coin are only about 15 mm tall. Always print one
test coin and look at it in daylight before sending the final proof for large orders.

## 4. Proof and approval

- Send `proof_for_customer.jpg` on WhatsApp.
- Get a written "APPROVE" (a WhatsApp reply is fine) plus **a 50% advance** before batch printing.
- Change the order status in the **Orders** tab: *proof sent → approved → advance paid*.

## 5. Printing (resin MSLA printer)

- Open the STL in the slicer (Chitubox / Lychee).
- **Orientation:** print coins **flat on the build plate**, back side down, no supports.
  Coins are only 3–5 mm tall, so they print in well under an hour, however many fit on the plate.
- Use the slicer's *Array/Duplicate* function to fill the plate. The number that fits
  (about 20 for 50 mm coins on a ~22×13 cm plate) goes in `config/pricing.json`
  (`coins_per_print_plate`).
- Layer height 0.05 mm. Use a tough or ABS-like resin so coins don't chip.
- **For 150+ pieces:** print a perfect master, make a silicone mould and cast in resin
  mixed with brass or bronze powder ("cold cast"). This gives real metal weight and shine, costs
  less per piece and frees the printer.
- **For 500+ pieces or real metal:** send the STL to a medal/trophy manufacturer for
  zinc-alloy die-casting.

## 6. Post-processing and finishing

1. Wash in IPA (2 × 3 min), dry, then UV cure (3–5 min each side).
2. Antique brass finish: spray a dark base (black/brown), dry-brush metallic gold on the
   raised parts, and seal with a clear matt or gloss lacquer.
3. Cold cast: buff with steel wool (#0000) to bring out the metal, then patinate lightly.

## 7. QC checklist (every 20th piece, plus every piece for the first batch)

- [ ] Names spelled correctly, date correct
- [ ] Faces recognisable, no blobs or holes
- [ ] No sticky resin, fully cured
- [ ] Finish even, no fingerprints
- [ ] Hole clean (if used), ribbon passes through
- [ ] Count matches the order (+3% spare)

## 8. Pack and deliver

Pouch or box, with a thank-you card with your Instagram handle and QR code (every guest
at the wedding is a potential customer). Collect the balance on delivery. Status → *delivered*.

## Selling it

- Partners: wedding planners, invitation card printers, return-gift shops, photographers.
  Offer a 10% referral fee.
- Keep a sample tray of 5 finished coins in different finishes, and always carry it.
- Offer upsells: a large (100–150 mm) framed version for the couple, and keychain versions.
