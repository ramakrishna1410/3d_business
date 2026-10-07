# Face Rakhi

The brother's **3D face** (the same Tripo head model as Royal Chess and the Face Pendant) in a
rakhi. There is an optional **lumba** with the bhabhi's face for a *Bhaiya + Bhabhi* set.
After Raksha Bandhan the thread slides out and the rakhi becomes a **keychain or pendant**
(top loop), so it is kept instead of thrown away.

## In the app (🪢 Face Rakhi tab)

1. **Head from**: Tripo head model file, or a photo (Tripo API credits; the photo is checked
   first).
2. **Order**: *Rakhi only*, or *Bhaiya + Bhabhi set*. The set adds a lumba with the second face.
3. **Design**:
   - *Kundan medallion* (32 mm): a ring of 16 stones, with the text under the face (BHAIYA or
     a name, up to about 8 letters).
   - *Flower* (40 mm): 12 petals, each with a stone.
4. **Made in**:
   - *Resin (painted gold)*: sharpest face. Sells for about ₹499 (set about ₹899).
   - *FDM (silk gold PLA)*: comes out gold, no painting. 20% bigger with deeper relief so the
     face reads through the layers. Sells for about ₹299–349.
5. **Stones**: red & green, red, blue & white or pearl white. The lumba uses pearls with
   red & green. This changes the preview; the STL has empty seats.
6. **Face size & position** if needed. Press **Preview** (free, about 5 s per piece), then
   **Make rakhi** (about 25 s for a set).

The output is an STL per piece, a front preview on the thread, a back view, making notes and a
quote.

## What is built in

- **Stone seats**: flat-bottom cups for flat-back stones (3 mm on the rakhi, 2 mm on the lumba),
  0.6 mm deep on small raised bezels. Glue them in with E6000 or Fevikwik gel.
- **Thread tunnel**: a 2 mm tunnel across a ridge on the back of the rakhi. The dori slides
  through, with no sewing or glue.
- **Top loop**: for a keyring after the festival, or for hanging the lumba from its bangle or
  chain.
- The face uses the same sharp relief, pupil dots and automatic chin detection as the pendant.
- STLs are closed solids that pass print-shop "non-manifold" checks.

## Printing

- **Resin**: a fine-detail resin (LEDO 6060 or grey), 0.025–0.05 mm layers. Supports on the
  back and loop only, never on the face. Paint gold, add a darker wash in the background, and
  dry-brush the face and rim.
- **FDM**: silk gold PLA, 0.2 mm nozzle, 0.06–0.08 mm layers. Face up with tree supports under
  the back ridge, or stood on its edge with a brim (try both on the first test).

## Selling notes

- Raksha Bandhan is in August. Start Instagram posts 6–8 weeks before. Rakhis mailed to
  brothers in other cities are a big market.
- Stop taking orders **10–12 days before** the festival (face model, print, paint, ship).
- The same pieces work for **Bhai Dooj** (after Diwali).
- Prices live in `config/pricing.json` → `rakhi`.
