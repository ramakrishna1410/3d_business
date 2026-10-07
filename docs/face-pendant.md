# Face Pendant

A person's 3D head (the same Tripo model used for Royal Chess) becomes a **raised portrait
pendant**. A jeweller prints it in castable resin and casts it in brass or silver. Popular as a
**his & hers pair**: the husband wears his wife's face and the wife wears his, with the same
names and date engraved on the back.

## In the app (💎 Face Pendant tab)

1. **Head from**: upload the Tripo head model (GLB/OBJ/STL), or a photo (Tripo API credits;
   the photo is checked first, so bad photos never cost credits). A head already made for a
   chess piece can be reused, with no new credits.
2. **Pair**: tick *His & hers pair* and add the second person's head. Each pendant is labelled
   with whose face it carries.
3. **Shape**: Round, Heart or Oval. **Size**: Small 20 mm, Medium 24 mm (default) or
   Large 28 mm. The oval is 22 × 30 mm at Medium.
4. **Portrait**: *Head & shoulders* (classic medallion) or *Face only* (bigger face).
5. **Rim**: Plain polished, or Beaded (traditional).
6. **Back**:
   - *Engraved*: up to 3 lines (names, message, date) in the 3 shop fonts. Type `♥` for a
     small engraved heart. Text that will not fit stops before anything is built.
   - *Hollow*: metal is removed behind the face (lighter, cheaper), with no text.
7. **Metal**: gold-plated brass, antique gold brass, rhodium-plated brass (silver look) or 925
   silver. This changes the preview colour, the weight and the quote.

The result is a front and back preview for the customer, one STL plus jeweller notes per
pendant, and a quote. It takes about 10–20 s per pendant.

## What the engine does automatically

- Finds the face direction, the crown and the **chin**, so every face is sized the same way.
- Turns the 3D face into a soft bas-relief (1.3 mm at 24 mm) that casts and polishes well. The
  bust fades into the background instead of ending in a ledge.
- Adds **pupil dots** (tiny dimples) when both eyes are found confidently. Tripo eyes are
  blank, and the dots make small faces look alive in metal. This can be switched off.
- Builds a **cast-in loop** (2.3 mm hole) at the top. The jeweller adds a jump ring for the chain.
- Engraves the back text 0.3 mm deep, **mirrored** so that it reads correctly on the back.
- Checks that nothing is thinner than 0.6–0.7 mm, and that the STL is one closed solid that
  passes print-shop "non-manifold" checks.

## Sending it to the casting shop

Send the STL and the `jeweller_notes.txt`. Ask for:

> Castable-resin print + casting in (brass / 925 silver), polish, (nickel-free gold plating 2–3
> micron + lacquer / antique finish / oxidised). Quote for 1 piece and for 10.

- **Only jewellery CAM printers** (castable resin, 25–50 micron) can print this. A normal FDM
  print shop cannot. An FDM print is fine only as a ₹30 size check on a chain.
- **Test first**: cast one piece in brass before offering silver.
- Ask for the sprue on the back edge, never on the face.
- For 925 silver, ask them to stamp "925" on the back.

## Prices

Edit `config/pricing.json` → `pendant`:

- **metal rate**: update the silver rate on the day you quote;
- **CAM print, casting per gram, plating**: your casting shop's quote;
- **packaging, labour and margin**.

The quote counts metal weight from the real model volume, with 15% casting loss. For a his &
hers pair the quantity is 2.
