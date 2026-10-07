"""Per-piece cost and suggested price, driven by config/pricing.json."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import load_pricing


@dataclass
class Quote:
    lines: dict[str, float] = field(default_factory=dict)
    cost_per_piece: float = 0.0
    price_per_piece: float = 0.0
    quantity: int = 1
    setup_fee: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def total_price(self) -> float:
        return self.price_per_piece * self.quantity + self.setup_fee

    def as_markdown(self, gst_percent: float) -> str:
        out = ["| Cost item (per piece) | ₹ |", "|---|---:|"]
        out += [f"| {k} | {v:,.1f} |" for k, v in self.lines.items()]
        out.append(f"| **Cost per piece** | **{self.cost_per_piece:,.1f}** |")
        out.append("")
        out.append(f"**Suggested price:** ₹{self.price_per_piece:,.0f} per piece × "
                   f"{self.quantity}" + (f" + ₹{self.setup_fee:,.0f} design fee"
                                         if self.setup_fee else "")
                   + f" = **₹{self.total_price:,.0f}** (+{gst_percent:.0f}% GST if registered)")
        profit = (self.price_per_piece - self.cost_per_piece) * self.quantity + self.setup_fee
        out.append(f"\nEstimated gross profit on this order: **₹{profit:,.0f}**")
        out += [f"\n- {n}" for n in self.notes]
        return "\n".join(out)


def _price_from_margin(cost: float, margin_percent: float) -> float:
    price = cost / max(1e-6, 1 - margin_percent / 100)
    # Round up to a "retail looking" number: ...9 for small, ...49/99 for bigger.
    if price < 100:
        return math.ceil(price / 5) * 5 - 1 if price > 5 else price
    step = 50 if price < 2000 else 100
    return math.ceil(price / step) * step - 1


def resin_cost(volume_mm3: float, p: dict) -> float:
    ml = volume_mm3 / 1000.0 * p["resin_waste_factor"] + p["support_and_raft_ml_per_piece"]
    return ml * p["resin_per_litre"] / 1000.0


def quote_medallion(volume_mm3: float, quantity: int, finish: str, packaging: str) -> Quote:
    p = load_pricing()
    m = p["medallion"]
    q = Quote(quantity=max(1, int(quantity)))
    q.lines["Resin"] = resin_cost(volume_mm3, p)
    plates = math.ceil(q.quantity / m["coins_per_print_plate"])
    q.lines["Printer time"] = plates * m["print_hours_per_plate"] * p["machine_cost_per_print_hour"] / q.quantity
    q.lines["Finish"] = m["finish_per_piece"].get(finish, 0)
    q.lines["Packaging"] = m["packaging_per_piece"].get(packaging, 0)
    q.lines["Labour (wash, cure, finish, pack)"] = m["labour_minutes_per_piece"] / 60 * p["labour_per_hour"]
    q.lines["Electricity & consumables"] = p["electricity_and_consumables_per_piece"]
    q.cost_per_piece = sum(q.lines.values())
    q.price_per_piece = _price_from_margin(q.cost_per_piece, m["target_margin_percent"])
    q.setup_fee = m["design_setup_fee"]
    hours = plates * m["print_hours_per_plate"]
    q.notes.append(f"{plates} print plate(s) ≈ {hours:.1f} printer hours "
                   f"(≈{math.ceil(hours / 8)} working day(s) on one printer).")
    if finish == "cold cast brass" and q.quantity >= 100:
        q.notes.append("For 100+ pieces consider a silicone mould of one printed master.")
    return q


def quote_drawing(volume_mm3: float, height_mm: float, paint: str, packaging: str) -> Quote:
    p = load_pricing()
    d = p["drawing"]
    q = Quote(quantity=1)
    q.lines["Resin"] = resin_cost(volume_mm3, p)
    hours = max(0.5, height_mm / 10 * d["print_hours_per_cm_height"])
    q.lines["Printer time"] = hours * p["machine_cost_per_print_hour"]
    q.lines["Paint"] = d["paint_per_piece"].get(paint, 0)
    q.lines["Packaging"] = d["packaging_per_piece"].get(packaging, 0)
    labour_min = d["labour_minutes"].get(paint, 30) + d["design_minutes"]
    q.lines["Labour (design check, post-process, paint)"] = labour_min / 60 * p["labour_per_hour"]
    q.lines["Electricity & consumables"] = p["electricity_and_consumables_per_piece"]
    q.cost_per_piece = sum(q.lines.values())
    q.price_per_piece = _price_from_margin(q.cost_per_piece, d["target_margin_percent"])
    q.notes.append(f"Estimated print time ≈ {hours:.1f} h (several pieces can share one plate).")
    return q


def quote_royal(volume_mm3: float, height_mm: float, finish: str, packaging: str) -> Quote:
    p = load_pricing()
    r = p["royal_chess"]
    q = Quote(quantity=1)
    # Resin vendors hollow busts this size; solid volume x factor is the resin actually used.
    q.lines["Resin (hollowed)"] = resin_cost(volume_mm3 * r["hollow_factor"], p)
    hours = max(1.0, height_mm / 10 * r["print_hours_per_cm_height"])
    q.lines["Printer time"] = hours * p["machine_cost_per_print_hour"]
    q.lines["3D head (Tripo credits)"] = r["head_model_cost"]
    q.lines["Finish"] = r["finish_per_piece"].get(finish, 0)
    q.lines["Packaging"] = r["packaging_per_piece"].get(packaging, 0)
    q.lines["Labour (check, support removal, cure, finish)"] = r["labour_minutes"] / 60 * p["labour_per_hour"]
    q.lines["Electricity & consumables"] = p["electricity_and_consumables_per_piece"]
    q.cost_per_piece = sum(q.lines.values())
    q.price_per_piece = _price_from_margin(q.cost_per_piece, r["target_margin_percent"])
    q.notes.append(f"Estimated print time ≈ {hours:.1f} h. Ask the print shop to hollow the piece "
                   "with drain holes (cheaper, less resin); add weight inside afterwards.")
    q.notes.append("If you outsource printing, replace resin + printer time with the shop's quote.")
    return q


def quote_pendant(volume_mm3: float, metal: str, quantity: int, packaging: str) -> Quote:
    """One cast pendant (per piece); quantity = pendants in the order (2 for a his & hers pair)."""
    p = load_pricing()
    c = p["pendant"]
    q = Quote(quantity=max(1, int(quantity)))
    grams = volume_mm3 / 1000 * c["density_g_per_cm3"][metal] * c["metal_loss_factor"]
    q.lines[f"Metal ({grams:.1f} g {metal})"] = grams * c["metal_rate_per_gram"][metal]
    q.lines["Castable-resin print (jeweller CAM)"] = c["cam_print_per_piece"]
    q.lines["Casting + polish"] = max(c["min_casting_charge"], grams * c["casting_per_gram"][metal])
    q.lines["Plating / finish"] = c["plating_per_piece"][metal]
    q.lines["3D head (Tripo credits)"] = c["head_model_cost"]
    q.lines["Packaging"] = c["packaging_per_piece"].get(packaging, 0)
    q.lines["Labour (files, checks, chain, packing)"] = c["labour_minutes"] / 60 * p["labour_per_hour"]
    q.cost_per_piece = sum(q.lines.values())
    q.price_per_piece = _price_from_margin(q.cost_per_piece, c["target_margin_percent"])
    q.notes.append("Jeweller prices are starting assumptions - update config/pricing.json with real quotes.")
    if metal == "925 Silver":
        q.notes.append("Silver rate changes daily: update metal_rate_per_gram before quoting.")
    q.notes.append("GST: 3% on silver metal value + 5% on making charges if you bill as jewellery.")
    return q
