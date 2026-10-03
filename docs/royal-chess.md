# Royal Chess - your face as the King, Queen or Bishop

A person's **3D head** becomes a chess piece: a portrait bust with a crown
(King), tiara (Queen) or mitre (Bishop - the kids' piece) on a classic chess
pedestal. Single colour - finished in bronze, gold or statue white.

```
photo ──(Tripo)──> 3D head ──(app: Royal Chess tab)──> King / Queen / Bishop STL
                                                       + preview + quote
```

## 1. Get the 3D head

The head must have the details **carved into the shape** (beard, eyebrows, eyes,
hair). A single-colour print loses everything that is only painted on.

### a) Tripo app / website (manual, works today)

1. **Image to 3D**, choose the template **3D Print** (not "3D Figurine" - it
   turns people into cartoons; plain "no template" keeps the beard and eyes only
   as colour, not as shape).
2. Use one **clear front photo**: face straight to the camera, no cap or
   sunglasses, good light, plain background.
3. Check it in **Solid view** (that is the shape that gets printed). The beard and
   eyebrows must still be visible there.
4. **Export as STL** (or GLB/OBJ). If the file is too big, reduce the face count in
   the export settings - 300k-500k faces is plenty (the head is only ~30 mm tall).

### b) Tripo API (automatic, from the app)

Choose **"Photo -> Tripo API"** in the tab. The app uploads the photo, asks for
*Ultra geometry, no texture* (`geometry_quality: detailed`, `texture: false`,
`pbr: false`, model `v3.1`), waits for the model and downloads it.

* API key: create one on Tripo's **API platform** (the API has its own credits,
  separate from the app subscription). Put it in a file called `.env` next to
  `app.py`:
  ```
  TRIPO_API_KEY=sk-...
  ```
  or paste it in **Settings -> Tripo API key** (kept in memory only).
  **Never commit the key** - `.env` is git-ignored.
* Cost: shown in the log after each run (`credits_consumed`).
* The API has no "3D Print" template. Test once with your own photo and compare
  with the app result; if the API head is too smooth, use way (a).

## 2. Make the piece (Royal Chess tab)

| Setting | Use |
|---|---|
| Piece | King (dad) · Queen (mom) · Bishop (child) |
| Size | chess set (King ~80 mm) · couple gift (King ~113 mm) |
| Quality | **preview** (~15 s) to check · **final** (2-6 min, 0.1 mm detail, needs ~4 GB free RAM) for printing |
| Adjust | only if needed: turn the head, move the neck cut, move the crown up/down |

The app automatically turns the head upright and facing front, cuts it at the neck
(keeping a beard), turns the AI model (often thousands of loose bits) into one
solid, fits the crown/tiara/mitre to the actual head shape, adds the royal bust
(ermine collar + chain, pearls, or a simple collar + cross) and the pedestal, and
writes **one watertight STL**.

Outputs in the order folder: `..._king.stl`, preview PNG, 3D preview, the head
model, `order.json` with the quote.

## 3. Print and finish

* **Resin (SLA/MSLA) only** - the beard and eyes are 0.1-0.3 mm details.
* Ask the print shop to **hollow it with drain holes** (about half the resin).
  Add weight inside (steel shot) and a felt pad under the base afterwards.
* Finishes: **bronze** or **gold** = dark base coat, metallic paint, lightly
  rubbed back so the grooves stay dark (this shows the face best); **statue
  white** = matt or satin white, not glossy.

## Privacy

Customer photos and head models stay in `orders/` (git-ignored). The repository
is public - never commit faces, heads or keys. With the API, the customer's photo
is sent to Tripo: tick the consent box only when the customer agreed.
