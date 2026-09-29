# Memory Factory: Phase 1

Personalised 3D keepsakes, made in Chennai. Phase 1 has two products that share one
automated software pipeline and one resin 3D printer.

| Product | Customer | Why it comes first |
|---|---|---|
| **Wedding return-gift medallions**: the couple's faces in relief on a coin with names and date | Families planning weddings, receptions, 60th birthdays and housewarmings, reached through wedding planners and invitation printers | Bulk orders of 100–500 pieces, so one order can equal a month of walk-in sales |
| **Kids' drawing → 3D sculpture**: a child's drawing turned into a puffed-up plaque or a standing figure with their name | Parents and grandparents; birthday gifts; schools (annual day) | No Indian service found; emotional, shareable, easy to get right technically |

Why these two: they reuse the same relief engine, need about ₹1–1.5 lakh of equipment
(a resin printer, a wash & cure station, paints) instead of a full studio, and can be
tested through Instagram, wedding expos and school tie-ups before renting mall space.

---

## The software: upload → finished 3D file, automatically

```
 UPLOAD ─► AUTO PIPELINE ──────────────────────────────────────────► OUTPUT
 couple photo   background removal → face detection & head-and-       STL for the printer
                shoulders crop → depth → bas-relief compression →     proof card for WhatsApp
                coin layout (big portrait, names curved on the rim)   3D preview, cost & price
                → watertight mesh
 kids drawing   find drawing on paper → inflate shape → pencil-line   STL (plaque / figure)
                grooves → name plate → watertight mesh                painted preview
                (or: AI full 3D via Meshy / import from Tripo)        painting guide, cost & price
```

Every job is saved as an order folder in `orders/` with the original image, the STL,
the previews and an `order.json`. The Orders tab tracks each order's status (proof sent,
approved, printing, … delivered).

### Quick start (Windows laptop)

1. Install **Python 3.11** from python.org and tick *"Add python.exe to PATH"*.
2. Download this repository (green **Code** button → *Download ZIP*) and unzip it.
3. Double-click **`start.bat`**. The first run installs packages (about 5 minutes); after
   that the app opens in your browser at http://127.0.0.1:7860.

On macOS/Linux, run `./start.sh` instead.

AI background removal (rembg) and face detection (OpenCV) are part of the standard install.
The background-removal model (~170 MB) downloads the first time you make a medallion.

**Rounder, more realistic faces (recommended once you're selling medallions):**
```
.venv\Scripts\activate
pip install -r requirements-ai.txt
```
This adds the Depth Anything V2 AI depth model. Without it, the app uses a simpler depth
estimate that still gives recognisable faces.
The **Settings** tab shows which engines are active.

**After `git pull`:** just run `start.bat` again. It installs any new packages automatically.

**Use it from a shop tablet or phone:** run `python app.py --lan` and open
`http://<laptop-ip>:7860` on a device on the same Wi-Fi.

### Command line (batch work)

```
python tools/medallion_relief.py couple.jpg --names "Ramesh & Meera" --date 22.04.2026 --qty 150 --hole
python tools/drawing_to_3d.py dino.jpg --name Aarav --line2 "Age 6 - 2026" --mode "standing figure"
```

### Tests

```
python -m pytest -q
```
The tests check that every generated model is closed (watertight) and printable, and that
the pricing is calculated.

---

## Repository map

| Path | What it is |
|---|---|
| `app.py` | The studio app (Gradio web UI) |
| `memory_factory/medallion.py` | Wedding medallion pipeline |
| `memory_factory/drawing.py` | Kids' drawing pipeline (plaque, standing figure, AI, import) |
| `memory_factory/relief.py` | Bas-relief compression (the quality core) and shape inflation |
| `memory_factory/depth.py` | AI depth (Depth Anything V2) with an offline fallback |
| `memory_factory/mesh.py` | Height map → watertight STL, previews |
| `memory_factory/render.py` | Material renders, WhatsApp proof card, painting guide |
| `memory_factory/costing.py` + `config/pricing.json` | Cost per piece and suggested price |
| `memory_factory/providers/meshy.py` | Optional Meshy API and import of Tripo/Meshy downloads |
| `tools/` | Command-line versions of both pipelines |
| `docs/wedding-medallion-workflow.md` | Medallion SOP: order → proof → print → pack |
| `docs/kids-drawing-workflow.md` | Drawing SOP: capture → 3D → print → paint |
| `docs/costing.md` | Cost model, pricing and break-even |

## Before you sell

- **Replace the numbers in `config/pricing.json`** with real supplier quotes (resin, paint,
  packaging, your labour rate). The quotes in the app are only as good as these numbers.
- **Consent:** the app will not generate anything until the customer-consent box is ticked.
  Order folders contain personal photos, so they are excluded from git (`.gitignore`). Back
  them up privately and delete them after an agreed period (for example, 90 days).
- **Cloud AI (Meshy/Tripo):** check the plan's commercial-use terms. Free tiers may make your
  models public. Tell customers when their image is sent to a cloud service.
- **Print a test piece** of every new design size before promising it to a customer.
