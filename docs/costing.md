# Costing & Pricing: Phase 1

> All numbers below are **starting assumptions** for planning. Replace them with real quotes
> from Chennai suppliers, update `config/pricing.json`, and the app's quotes follow
> automatically. Prices are in INR, excluding GST.

## 1. One-time setup (approximate)

| Item | Estimate (₹) | Notes |
|---|---:|---|
| Resin MSLA printer (~8–10" plate, 12K+) | 40,000 – 1,00,000 | Bigger plate = more coins per print |
| Wash & cure station | 10,000 – 20,000 | |
| Resin (5 L to start), IPA, gloves, masks | 15,000 | Resin fumes: ventilate, use an extractor fan |
| Paints, brushes, primer, varnish, airbrush (optional) | 5,000 – 20,000 | |
| Silicone and casting materials for moulds (medallions) | 5,000 – 10,000 | |
| Sample stock for marketing (30 coins, 10 drawings) | 5,000 | |
| Laptop | already owned | The app runs on a normal laptop |
| **Total** | **≈ 0.8 – 1.8 lakh** | Compare with ₹5–10 lakh+ for the full studio idea |

## 2. Per-piece cost model (as used by the app)

```
resin cost      = model volume (from the STL) × waste factor × resin ₹/ml  + supports
printer cost    = print hours × ₹/hour (machine wear, screen/FEP replacement)
labour          = minutes × your hourly rate
+ finish/paint + packaging + electricity & consumables
─────────────────────────────────────────────
cost per piece  → price = cost ÷ (1 − target margin), rounded to a retail number
```

The resin volume is calculated exactly from each generated model, so a bigger coin or a
thicker figure is priced correctly without guessing.

### Example: 50 mm medallion, 150 pieces, antique brass paint, velvet pouch

| Item | ₹ / piece |
|---|---:|
| Resin (~5.5 ml incl. waste) | 17 |
| Printer time (8 plates × 1.2 h) | 2 |
| Finish | 12 |
| Pouch | 15 |
| Labour (3 min) | 10 |
| Electricity & consumables | 5 |
| **Cost** | **≈ 61** |
| **Suggested price (55% margin)** | **₹149** + ₹1,500 design fee |

**Order value ≈ ₹23,850 → gross profit ≈ ₹14,700.**
For comparison, typical return gifts cost ₹50–300 each, so ₹149 for a personalised
metal-look coin is competitive. Test ₹129, ₹149 and ₹199 with real customers.

### Example: kids' drawing, 100 mm, hand-painted, display box

| Item | ₹ |
|---|---:|
| Resin (~45–55 ml) | 125–150 |
| Printer time (~2 h) | 50 |
| Paint | 60 |
| Display box | 90 |
| Labour (15 min design + 75 min painting) | 300 |
| Electricity & consumables | 5 |
| **Cost** | **≈ 630–660** |
| **Suggested price (60% margin)** | **≈ ₹1,599** |

For comparison, Chennai 3D figurine studios charge ₹3,999+ with a 2–3 week wait. There is room
to price the painted standing figure at ₹1,999–2,499 once samples look good. Labour is the
biggest cost, so painting speed decides profit.

## 3. Monthly break-even

| Fixed cost per month | ₹ (assumption) |
|---|---:|
| Workspace (home or small room, no mall) | 0 – 10,000 |
| Instagram / Meta ads | 5,000 – 10,000 |
| Software (Meshy/Tripo plan, optional) | 1,500 – 3,000 |
| Printer maintenance (screen, FEP) | 1,000 |
| **Total** | **≈ 7,500 – 24,000** |

Gross profit per unit of sales:
- one 150-piece medallion order ≈ **₹14,700**
- one drawing sculpture ≈ **₹950**

**Break-even at ₹24,000/month:** about 2 medallion orders, **or** 1 medallion order + 10
drawings, **or** 25 drawings. That is a realistic target for Phase 1 without a mall kiosk.

## 4. Capacity check (one printer)

- Medallions: ~20 coins per plate, ~1.2 h per plate, 6 plates in a working day ≈ **120 coins a day**.
  The real limit is finishing and packing labour, about 3 min per coin (≈ 150/day for one person).
- Drawings: the printer is not the limit. **Painting is** (4–6 pieces a day per painter).

When capacity is consistently above 70% for a month, buy a second printer or hire a painter.

## 5. Things to measure in the first month

1. Real resin used per coin and per figure (weigh the bottle before and after a batch)
2. Real print time per plate
3. Real minutes of finishing and painting per piece
4. Failure rate (reprints). Add it to `resin_waste_factor`
5. Conversion: enquiries → proofs → paid orders

Update `config/pricing.json` with these numbers. The app's quotes then reflect reality.
