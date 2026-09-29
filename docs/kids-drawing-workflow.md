# Kids' Drawing → 3D Sculpture: Workflow (SOP)

**Product:** a child's own drawing turned into a 3D keepsake, hand-painted to match the
original colours, with the child's name and age.

| Mode in the app | What it makes | Best for | Needs |
|---|---|---|---|
| **relief plaque** | The drawing puffed up on a rounded plaque with the name underneath (wall or desk) | Any drawing; the strongest and cheapest option | Nothing extra (offline) |
| **standing figure** | The drawing's outline inflated on both sides, standing on a name base | Animals, monsters, people with a clear outline | Nothing extra (offline) |
| **line-art relief** | Outline/pencil sketches (e.g. a Ganesha on a peepal leaf, kolam, a temple). The plaque takes the drawing's **own outline shape**; the figure is raised above the background, each part gently rounded, pencil lines pressed in | Devotional art, wall plaques, sketches with a closed outer outline | Nothing extra (offline) |
| **ai full 3d** | A real 3D model of the character, made automatically by Meshy | Premium orders | Meshy API key and paid credits |
| **import 3d model** | You make the model on the Tripo or Meshy website, download GLB/OBJ/STL, and the app scales it, cleans it and adds the name base | Premium orders without an API key | A Tripo/Meshy account |

```
DRAWING PHOTO ─► APP (auto 3D) ─► PREVIEW TO PARENT ─► PAYMENT ─► PRINT
    ─► WASH & CURE ─► PAINT (using the painting guide) ─► BOX WITH ORIGINAL ─► HANDOVER
```

## 1. Capturing the drawing

- Put the drawing flat on a table in daylight, with no shadows (don't stand between the light
  and the paper).
- Take the photo from directly above, filling the frame with the paper.
- A scan is even better.
- Drawings need a **clear outline**. Very light pencil sketches can be traced over with a
  marker, with the parent's permission (or on a photocopy).

## 2. Automatic 3D (app → 🖍️ Kids Drawing 3D)

1. Upload the drawing, enter the child's name and second line (age/year), and pick the product mode.
2. Choose paint and packaging, fill in the parent's details and **tick consent**.
3. Click **Generate 3D**. You get:
   - a painted preview to show the parent
   - the STL file for the printer
   - `painting_guide.png`: the main colours with hex codes and how much of each is used
   - the cost breakdown and suggested price
4. Adjust (*Design settings*):
   | Problem | Fix |
   |---|---|
   | Looks too flat | *Puffiness* 8–12 mm |
   | Thin parts (legs, tails) too weak | Increase *Size*, or use *relief plaque* instead of *standing figure* |
   | Pencil lines lost | *Line groove depth* 0.8–1.0 mm |
   | Paper edge or shadow included | Retake the photo with even light, or crop tighter before uploading |

### Using Tripo or Meshy manually ("import 3d model")

1. On the tool's website, upload the drawing and generate the model. Tip: first ask an AI image
   tool for a *"3D cartoon character render of this drawing, front view, plain background,
   keep the child's shapes and colours"*, then use that image. This gives much better models.
2. Download as **GLB** (or OBJ/STL).
3. In the app, choose **import 3d model**, upload the drawing (for the preview and palette)
   and the downloaded model, then click **Generate 3D**.
4. The app merges vertices, fills holes, removes floating crumbs, scales the model to the size
   you set and adds the name base. The log tells you if the model is still not watertight;
   in that case use the slicer's auto-repair.
5. **Check the tool's commercial-use terms.** Free plans may make your models public.

## 3. Printing

- **Relief plaque:** print flat, back down, no supports needed. Around 1–2 hours.
- **Standing figure / AI model:** print upright on the base. Add light supports under
  overhangs such as tails and arms in the slicer, and hollow large models (2 mm wall with
  drain holes) to save resin.
- Use a tough resin for anything a child will handle.

## 4. Painting

1. Wash, cure and remove supports. Sand the support marks lightly.
2. Prime with a white spray primer.
3. Paint the big areas first using the colours in `painting_guide.png` (acrylics).
4. Pick out the grooves (the child's pencil lines) with a thin black liner. This is what
   makes it look like *their* drawing.
5. Seal with matt varnish.

**Time target:** 45–75 minutes of painting per piece. Record your actual times and update
`labour_minutes` in `config/pricing.json`.

## 5. Packaging and handover

- A display box with a window, with a printed copy of the original drawing next to the model.
  Parents love seeing the before and after.
- Photograph every finished piece next to the drawing (with the parent's permission) for Instagram.

## Selling it

- **Schools:** an annual-day or art-week offer where every child's drawing can be ordered, with a
  small fundraising share for the school.
- **Birthday parties:** a drawing station at the party, with the models delivered a week later.
- **Grandparents:** make it a gift ("from your grandchild's imagination").
- Instagram Reels showing drawing → 3D preview → finished model are your best advertising.

## Line-art relief: tips

- The **outer outline must be one closed line**. It becomes the plaque's edge.
- Areas that **touch the outer outline** become the background layer (the leaf); areas
  **inside** become the raised figure (the Ganesha). Small gaps in pencil lines are bridged
  automatically.
- Darker pencil = deeper groove, so shading (for example, hair) shows up as texture. Very
  faint lines may disappear; go over them with a darker pencil or pen first.
- Photograph from directly above in daylight, with no fingers or shadows on the paper.
- The name fields are not used in this mode (it's an art piece). The **Size** slider sets
  the longest side; 120–150 mm works well for wall plaques.
- Print it flat, back down. At 150 mm it's about 8 mm thick, uses ~70 ml of resin and
  prints in about 1.5–2 hours. Paint it antique gold or brass, or cast it from a mould for a
  metal look.
