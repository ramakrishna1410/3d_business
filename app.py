"""Memory Factory studio app.

Run:   python app.py            (opens http://127.0.0.1:7860 in your browser)
       python app.py --lan      (also reachable from a tablet/phone on the shop Wi-Fi)

Tabs
    1. Wedding Medallion   couple photo  -> coin STL + WhatsApp proof + quote
    2. Kids Drawing 3D     drawing photo -> plaque / figure STL + painting guide + quote
    3. Royal Chess         photo (Tripo API) or Tripo 3D head -> King/Queen/Bishop STL
    4. Face Pendant        Tripo 3D head -> cast pendant STL (jeweller) + front/back proof
    5. Orders              every job saved in ./orders
    6. Settings            AI engine status, API keys, prices
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import gradio as gr

from memory_factory import (depth, drawing, engrave, imaging, medallion, orders, pendant, photo_check,
                            royal)
from memory_factory.config import PRICING_FILE, load_local_env, load_pricing
from memory_factory.render import MATERIALS

MEDALLION_FINISHES = list(load_pricing()["medallion"]["finish_per_piece"])
MEDALLION_PACKAGING = list(load_pricing()["medallion"]["packaging_per_piece"])
DRAWING_PAINT = list(load_pricing()["drawing"]["paint_per_piece"])
DRAWING_PACKAGING = list(load_pricing()["drawing"]["packaging_per_piece"])
ROYAL_FINISHES = list(load_pricing()["royal_chess"]["finish_per_piece"])
ROYAL_PACKAGING = list(load_pricing()["royal_chess"]["packaging_per_piece"])
PENDANT_PACKAGING = list(load_pricing()["pendant"]["packaging_per_piece"])

load_local_env()
_session = {"meshy_key": os.environ.get("MESHY_API_KEY", ""),
            "tripo_key": os.environ.get("TRIPO_API_KEY", "")}


def _customer(name, phone, consent):
    if not consent:
        raise gr.Error("Please tick the customer consent box first (photo use & storage).")
    return {"name": name or "", "phone": phone or "", "consent": bool(consent)}


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def run_medallion(photo, names, date, extra, layout, diameter, relief_mm, base_mm, text_mm, hole,
                  remove_bg, engine, compression, detail, material, quantity, finish,
                  packaging, cust_name, cust_phone, consent, progress=gr.Progress()):
    if photo is None:
        raise gr.Error("Upload the couple's photo first.")
    customer = _customer(cust_name, cust_phone, consent)
    progress(0.1, desc="Removing background, estimating depth...")
    s = medallion.MedallionSettings(
        names=names, date=date, extra=extra, layout=layout, diameter_mm=diameter, relief_mm=relief_mm,
        base_mm=base_mm, text_mm=text_mm, keychain_hole=hole, remove_background=remove_bg,
        depth_engine=engine, compression=compression, detail=detail, material=material)
    try:
        r = medallion.generate(photo, s, customer, int(quantity), finish, packaging)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    progress(1.0)
    for w in r.warnings:
        gr.Warning(w)
    info = "\n".join(f"- {line}" for line in r.log)
    if r.warnings:
        info = "\n".join(f"> ⚠️ {w}" for w in r.warnings) + "\n\n" + info
    return (str(r.proof), str(r.glb) if r.glb else None, [str(r.stl), str(r.proof)],
            r.quote_md, f"**Order {r.order_id}** saved in `{r.folder}`\n\n{info}")


def run_drawing(image, child, age_line, mode, size_mm, puff, line_depth, roundness, model_file,
                paint, packaging, cust_name, cust_phone, consent, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload a photo/scan of the drawing first.")
    customer = _customer(cust_name, cust_phone, consent)
    s = drawing.DrawingSettings(child_name=child, age_line=age_line, mode=mode, size_mm=size_mm,
                                puff_mm=puff, line_depth_mm=line_depth, roundness=roundness)
    progress(0.1, desc=f"Building {mode}...")
    try:
        r = drawing.generate(image, s, customer, paint, packaging,
                             meshy_api_key=_session["meshy_key"] or None,
                             model_file=model_file, progress=progress)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    info = "\n".join(f"- {line}" for line in r.log)
    return (str(r.render), str(r.glb) if r.glb else None, str(r.painting_guide),
            [str(r.stl), str(r.render), str(r.painting_guide)], r.quote_md,
            f"**Order {r.order_id}** saved in `{r.folder}`\n\n{info}")


ROYAL_PIECES = {"King": "king", "Queen": "queen", "Bishop (child)": "bishop"}
ROYAL_SOURCES = ["Tripo 3D file (GLB/OBJ/STL)", "Photo -> Tripo API (automatic)"]


def _save_crop(check) -> str | None:
    if check.crop is None:
        return None
    path = Path(tempfile.mkdtemp(prefix="tripo_crop_")) / "for_tripo.jpg"
    check.crop.save(path, quality=95)
    return str(path)


def check_royal_photo(photo):
    """Free, offline check of the customer photo + the Tripo-ready close-up."""
    if not photo:
        raise gr.Error("Upload the customer's photo first.")
    check = photo_check.check_photo(photo)
    crop = _save_crop(check)
    return check.as_markdown(), crop, crop


def run_royal(source, model_file, photo, piece, style, size, quality, finish, packaging, turn,
              auto_neck, neck, crown, tidy, font, name, msg1, msg2, date, cust_name, cust_phone, consent,
              progress=gr.Progress()):
    customer = _customer(cust_name, cust_phone, consent)
    if source == ROYAL_SOURCES[1]:
        if not photo:
            raise gr.Error("Upload a clear front photo (face visible, no cap or sunglasses).")
        if not _session["tripo_key"]:
            raise gr.Error("Add your Tripo API key in the Settings tab first.")
        check = photo_check.check_photo(photo)        # never spend credits on a bad photo
        if not check.ok:
            raise gr.Error("Photo check failed - no Tripo credits used. "
                           + " ".join(check.messages))
        src = _save_crop(check) or photo              # send the face close-up, not the whole photo
    else:
        if not model_file:
            raise gr.Error("Upload the head model exported from Tripo (GLB, OBJ or STL).")
        src = model_file
    s = royal.RoyalSettings(piece=ROYAL_PIECES[piece], style=style, size=size, quality=quality,
                            turn=turn,
                            neck=None if auto_neck else neck, crown=crown, tidy_hair=tidy, font=font,
                            name=name or "", message1=msg1 or "", message2=msg2 or "", date=date or "",
                            finish=finish,
                            packaging=packaging)
    progress(0.02, desc="Starting...")
    try:
        r = royal.generate(src, s, customer, api_key=_session["tripo_key"], progress=progress)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    info = "\n".join(f"- {line}" for line in r.log)
    files = ([str(r.stl), str(r.extra["print_notes"]), str(r.render)]
             + ([str(r.extra["underside"])] if "underside" in r.extra else [])
             + ([str(r.head_model)] if r.head_model else []))
    return (str(r.render), str(r.glb) if r.glb else None, files, r.quote_md,
            f"**Order {r.order_id}** saved in `{r.folder}`\n\n{info}")


def _pendant_source(source, model_file, photo, who):
    if source == ROYAL_SOURCES[1]:
        if not photo:
            raise gr.Error(f"{who}: upload a clear front photo (face visible, no cap or sunglasses).")
        if not _session["tripo_key"]:
            raise gr.Error("Add your Tripo API key in the Settings tab first.")
        check = photo_check.check_photo(photo)        # never spend credits on a bad photo
        if not check.ok:
            raise gr.Error(f"{who}: photo check failed - no Tripo credits used. " + " ".join(check.messages))
        return _save_crop(check) or photo
    if not model_file:
        raise gr.Error(f"{who}: upload the head model exported from Tripo (GLB, OBJ or STL).")
    return model_file


def run_pendant(source, model1, photo1, label1, pair, model2, photo2, label2, shape, size, crop, rim,
                detail, back, metal, font, line1, line2, date, pupils, turn, packaging, cust_name,
                cust_phone, consent, progress=gr.Progress()):
    customer = _customer(cust_name, cust_phone, consent)
    s = pendant.PendantSettings(shape=shape, size=size, crop=crop, rim=rim, detail=detail, back=back,
                                metal=metal,
                                font=font, line1=line1 or "", line2=line2 or "", date=date or "",
                                pupils=bool(pupils), turn=turn, packaging=packaging)
    try:
        pendant.check_text(s)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    sources = [(_pendant_source(source, model1, photo1, "Pendant 1"), (label1 or "Pendant 1").strip())]
    if pair:
        sources.append((_pendant_source(source, model2, photo2, "Pendant 2"), (label2 or "Pendant 2").strip()))
    progress(0.02, desc="Starting...")
    try:
        r = pendant.generate(sources, s, customer, api_key=_session["tripo_key"], progress=progress)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    info = "\n".join(f"- {line}" for line in r.log)
    return ([str(p) for p in r.previews], str(r.glb) if r.glb else None, [str(f) for f in r.files],
            r.quote_md, f"**Order {r.order_id}** saved in `{r.folder}`\n\n{info}")


ORDER_COLS = ["order_id", "date", "product", "customer", "phone", "qty", "price", "status", "folder"]
STATUSES = ["proof sent", "approved", "advance paid", "printing", "finishing", "ready",
            "delivered", "cancelled"]


def order_table():
    rows = orders.list_orders()
    return [[r[c] for c in ORDER_COLS] for r in rows]


def set_status(order_id, status):
    for r in orders.list_orders():
        if r["order_id"] == order_id.strip():
            path = os.path.join(r["folder"], "order.json")
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            data["status"] = status
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            return order_table(), f"Order {order_id} -> **{status}**"
    raise gr.Error(f"Order '{order_id}' not found.")


def engine_status():
    ok, no = "✅", "❌"
    return "\n".join([
        "| Engine | Status | What it does |", "|---|---|---|",
        f"| AI depth (Depth Anything V2) | {ok if depth.ai_depth_available() else no + ' not installed'} "
        "| Much better faces on medallions |",
        f"| AI background removal (rembg) | {ok if imaging.rembg_available() else no + ' not installed'} "
        "| Cuts people out of busy backgrounds |",
        f"| Meshy API key | {ok + ' set' if _session['meshy_key'] else no + ' not set'} "
        "| 'ai full 3d' drawing mode |",
        f"| Tripo API key | {ok + ' set' if _session['tripo_key'] else no + ' not set'} "
        "| Royal Chess: photo -> 3D head automatically |",
        "", "Install the AI engines with `pip install -r requirements-ai.txt` "
        "(see README). Everything works without them, just with simpler results.",
    ])


def save_key(key):
    _session["meshy_key"] = key.strip()
    return engine_status()


def save_tripo_key(key):
    _session["tripo_key"] = key.strip()
    return engine_status()


def load_prices():
    return PRICING_FILE.read_text(encoding="utf-8")


def save_prices(text):
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        raise gr.Error(f"Not valid JSON: {exc}") from exc
    PRICING_FILE.write_text(text, encoding="utf-8")
    return "Saved. New quotes will use these numbers."


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Memory Factory Studio") as ui:
        gr.Markdown("# Memory Factory Studio\nUpload → automatic 3D → proof for the customer → "
                    "STL ready for the printer.")

        with gr.Tab("💍 Wedding Medallion"):
            with gr.Row():
                with gr.Column(scale=1):
                    m_photo = gr.Image(label="Couple photo (plain background works best)",
                                       type="pil", height=320)
                    m_names = gr.Textbox(label="Names", value="Ramesh & Meera")
                    with gr.Row():
                        m_date = gr.Textbox(label="Date", value="22.04.2026")
                        m_extra = gr.Textbox(label="Message (optional, small)", value="",
                                             placeholder="e.g. With Love & Thanks")
                    m_layout = gr.Radio(medallion.LAYOUTS, value="coin", label="Layout",
                                        info="coin: big faces + names curved along the rim | "
                                             "classic: smaller photo + straight text")
                    with gr.Row():
                        m_qty = gr.Number(label="Quantity", value=150, precision=0)
                        m_finish = gr.Dropdown(MEDALLION_FINISHES, value=MEDALLION_FINISHES[1],
                                               label="Finish")
                        m_pack = gr.Dropdown(MEDALLION_PACKAGING, value=MEDALLION_PACKAGING[1],
                                             label="Packaging")
                    with gr.Accordion("Design settings", open=False):
                        m_diam = gr.Slider(30, 80, value=50, step=1, label="Diameter (mm)")
                        m_relief = gr.Slider(0.6, 3.0, value=1.5, step=0.1, label="Face relief height (mm)")
                        m_base = gr.Slider(1.5, 5, value=2.5, step=0.1, label="Coin thickness (mm)")
                        m_text = gr.Slider(0.3, 1.5, value=0.7, step=0.1, label="Letter height (mm)")
                        m_hole = gr.Checkbox(label="Keychain / ribbon hole", value=False)
                        m_bg = gr.Checkbox(label="Remove background", value=True)
                        m_engine = gr.Radio(["auto", "ai", "fast"], value="auto", label="Depth engine")
                        m_comp = gr.Slider(0, 1, value=0.45, step=0.05, label="Depth (flat ⟷ deep)")
                        m_detail = gr.Slider(0, 2, value=1.0, step=0.1, label="Fine detail")
                        m_mat = gr.Dropdown(list(MATERIALS), value="antique brass", label="Preview material")
                    with gr.Accordion("Customer", open=True):
                        m_cname = gr.Textbox(label="Customer name")
                        m_cphone = gr.Textbox(label="Phone / WhatsApp")
                        m_consent = gr.Checkbox(label="Customer agrees we store and use this photo "
                                                "only for their order")
                    m_go = gr.Button("Generate medallion", variant="primary")
                with gr.Column(scale=1):
                    m_proof = gr.Image(label="Proof for customer (send on WhatsApp)", height=420)
                    m_3d = gr.Model3D(label="3D preview", height=320)
                    m_files = gr.File(label="Downloads (STL for the printer)", file_count="multiple")
                    m_quote = gr.Markdown()
                    m_log = gr.Markdown()
            m_go.click(run_medallion,
                       [m_photo, m_names, m_date, m_extra, m_layout, m_diam, m_relief, m_base, m_text, m_hole,
                        m_bg, m_engine, m_comp, m_detail, m_mat, m_qty, m_finish, m_pack,
                        m_cname, m_cphone, m_consent],
                       [m_proof, m_3d, m_files, m_quote, m_log])

        with gr.Tab("🖍️ Kids Drawing 3D"):
            with gr.Row():
                with gr.Column(scale=1):
                    d_img = gr.Image(label="Photo/scan of the drawing (white paper, no shadows)",
                                     type="pil", height=320)
                    with gr.Row():
                        d_child = gr.Textbox(label="Child's name", value="Aarav")
                        d_age = gr.Textbox(label="Second line", value="Age 6 - 2026")
                    d_mode = gr.Radio(drawing.MODES, value="relief plaque", label="Product")
                    d_model = gr.File(label="Only for 'import 3d model': GLB/OBJ/STL downloaded "
                                            "from Tripo or Meshy", file_types=[".glb", ".obj", ".stl", ".ply"],
                                      type="filepath")
                    with gr.Row():
                        d_paint = gr.Dropdown(DRAWING_PAINT, value=DRAWING_PAINT[-1], label="Paint")
                        d_pack = gr.Dropdown(DRAWING_PACKAGING, value=DRAWING_PACKAGING[-1],
                                             label="Packaging")
                    with gr.Accordion("Design settings", open=False):
                        d_size = gr.Slider(50, 200, value=100, step=5, label="Size - longest side / height (mm)")
                        d_puff = gr.Slider(2, 15, value=6, step=0.5, label="Puffiness (mm)")
                        d_line = gr.Slider(0, 1.5, value=0.5, step=0.1, label="Line groove depth (mm)")
                        d_round = gr.Slider(0.5, 2, value=1.0, step=0.1, label="Roundness")
                    with gr.Accordion("Customer", open=True):
                        d_cname = gr.Textbox(label="Parent's name")
                        d_cphone = gr.Textbox(label="Phone / WhatsApp")
                        d_consent = gr.Checkbox(label="Parent agrees we store and use this drawing "
                                                "only for their order")
                    d_go = gr.Button("Generate 3D", variant="primary")
                with gr.Column(scale=1):
                    d_render = gr.Image(label="Painted preview", height=360)
                    d_3d = gr.Model3D(label="3D preview", height=320)
                    d_guide = gr.Image(label="Painting guide", height=160)
                    d_files = gr.File(label="Downloads (STL for the printer)", file_count="multiple")
                    d_quote = gr.Markdown()
                    d_log = gr.Markdown()
            d_go.click(run_drawing,
                       [d_img, d_child, d_age, d_mode, d_size, d_puff, d_line, d_round, d_model,
                        d_paint, d_pack, d_cname, d_cphone, d_consent],
                       [d_render, d_3d, d_guide, d_files, d_quote, d_log])

        with gr.Tab("♚ Royal Chess"):
            gr.Markdown("A person's **3D head** becomes a King, Queen or Bishop chess piece "
                        "(crown/tiara/mitre fitted to the head, royal bust, chess pedestal). "
                        "See `docs/royal-chess.md`.")
            with gr.Row():
                with gr.Column(scale=1):
                    r_source = gr.Radio(ROYAL_SOURCES, value=ROYAL_SOURCES[0], label="Head from")
                    r_model = gr.File(label="Head model from the Tripo app (template: 3D Print)",
                                      file_types=[".glb", ".gltf", ".obj", ".stl"], type="filepath")
                    r_photo = gr.Image(label="Or: clear front photo (uses Tripo API credits)",
                                       type="filepath", height=240)
                    with gr.Accordion("Check photo (free - do this before using Tripo)", open=False):
                        r_check_btn = gr.Button("Check photo + make Tripo close-up")
                        r_check = gr.Markdown()
                        with gr.Row():
                            r_crop = gr.Image(label="Tripo-ready close-up", height=220,
                                              interactive=False)
                            r_crop_file = gr.File(label="Download (upload this to the Tripo app)")
                    with gr.Row():
                        r_piece = gr.Radio(list(ROYAL_PIECES), value="King", label="Piece")
                        r_size = gr.Radio(list(royal.SIZES), value=list(royal.SIZES)[1], label="Size")
                    r_style = gr.Radio(royal.STYLES, value=royal.STYLES[0], label="Design",
                                       info="smooth statue: clean crown with a cross, plain mantle | "
                                            "classic royal: ermine collar, chain of office")
                    r_quality = gr.Radio(["preview", "final"], value="preview", label="Quality",
                                         info="preview ≈ 30-40 s to check the layout | final ≈ 5-8 min, "
                                              "full beard/eye detail for printing")
                    with gr.Row():
                        r_finish = gr.Dropdown(ROYAL_FINISHES, value="bronze", label="Finish")
                        r_pack = gr.Dropdown(ROYAL_PACKAGING, value="velvet box", label="Packaging")
                    with gr.Accordion("Adjust (only if the automatic result looks wrong)", open=False):
                        r_turn = gr.Radio(royal.TURNS, value="auto", label="Turn head (face direction)")
                        r_autoneck = gr.Checkbox(value=True, label="Automatic neck cut")
                        r_neck = gr.Slider(0.02, 0.5, value=0.15, step=0.01,
                                           label="Neck cut (fraction of model height)")
                        r_crown = gr.Slider(0.6, 0.95, value=0.77, step=0.01,
                                            label="Crown / tiara / mitre height on the head")
                        r_tidy = gr.Checkbox(value=True, label="Tidy hair (trim loose strands and long "
                                             "hair below the neck - fragile in resin)")
                    with gr.Accordion("Names & message (engraved, optional)", open=False):
                        r_font = gr.Radio(list(engrave.FONTS), value=list(engrave.FONTS)[0], label="Font")
                        r_name = gr.Textbox(label="Name on the front of the stand",
                                            info="up to ~8 letters (couple gift) / ~6 (chess set)",
                                            max_lines=1, max_length=14)
                        with gr.Row():
                            r_msg1 = gr.Textbox(label="Under the base - line 1", max_lines=1,
                                                max_length=20, placeholder="Ram & Meera")
                            r_msg2 = gr.Textbox(label="Under the base - line 2", max_lines=1,
                                                max_length=20, placeholder="Forever")
                        r_date = gr.Textbox(label="Date under the base", max_lines=1, max_length=14,
                                            placeholder="05.10.2026")
                    with gr.Accordion("Customer", open=True):
                        r_cname = gr.Textbox(label="Customer name")
                        r_cphone = gr.Textbox(label="Phone / WhatsApp")
                        r_consent = gr.Checkbox(label="Customer agrees we use this photo/model only for "
                                                "their order (and, for the API, that it is sent to Tripo)")
                    r_go = gr.Button("Make chess piece", variant="primary")
                with gr.Column(scale=1):
                    r_render = gr.Image(label="Preview", height=440)
                    r_3d = gr.Model3D(label="3D preview", height=360)
                    r_files = gr.File(label="Downloads (STL + print notes for the print shop)",
                                      file_count="multiple")
                    r_quote = gr.Markdown()
                    r_log = gr.Markdown()
            r_check_btn.click(check_royal_photo, r_photo, [r_check, r_crop, r_crop_file])
            r_go.click(run_royal,
                       [r_source, r_model, r_photo, r_piece, r_style, r_size, r_quality, r_finish, r_pack, r_turn,
                        r_autoneck, r_neck, r_crown, r_tidy, r_font, r_name, r_msg1, r_msg2,
                        r_date, r_cname, r_cphone, r_consent],
                       [r_render, r_3d, r_files, r_quote, r_log])

        with gr.Tab("💎 Face Pendant"):
            gr.Markdown("A person's **3D head** becomes a raised portrait pendant for **casting** "
                        "(jeweller's castable-resin print -> brass or silver). Names and date are "
                        "engraved on the back. See `docs/face-pendant.md`.")
            with gr.Row():
                with gr.Column(scale=1):
                    p_source = gr.Radio(ROYAL_SOURCES, value=ROYAL_SOURCES[0], label="Head from")
                    with gr.Row():
                        p_model1 = gr.File(label="Pendant 1: head model (GLB/OBJ/STL)",
                                           file_types=[".glb", ".gltf", ".obj", ".stl"], type="filepath")
                        p_photo1 = gr.Image(label="Or photo (Tripo API credits)", type="filepath", height=160)
                    p_label1 = gr.Textbox(label="Pendant 1: whose face", value="Meera", max_lines=1)
                    p_pair = gr.Checkbox(value=False, label="His & hers pair (second pendant with the "
                                         "other person's face)")
                    with gr.Row():
                        p_model2 = gr.File(label="Pendant 2: head model", type="filepath",
                                           file_types=[".glb", ".gltf", ".obj", ".stl"])
                        p_photo2 = gr.Image(label="Or photo", type="filepath", height=160)
                    p_label2 = gr.Textbox(label="Pendant 2: whose face", value="Ram", max_lines=1)
                    with gr.Row():
                        p_shape = gr.Radio(pendant.SHAPES, value="Round", label="Shape")
                        p_size = gr.Radio(list(pendant.SIZES), value="Medium 24 mm", label="Size")
                    with gr.Row():
                        p_crop = gr.Radio(pendant.CROPS, value=pendant.CROPS[0], label="Portrait")
                        p_rim = gr.Radio(pendant.RIMS, value="Plain", label="Rim")
                    p_detail = gr.Radio(pendant.DETAILS, value=pendant.DETAILS[0], label="Face detail",
                                        info="Sharp: crisp eyes, lips, beard and outline (best likeness) | "
                                             "Soft: gentle worn-coin look")
                    with gr.Row():
                        p_metal = gr.Dropdown(list(pendant.METALS), value="Gold-plated brass", label="Metal")
                        p_pack = gr.Dropdown(PENDANT_PACKAGING, value="velvet box", label="Packaging")
                    with gr.Accordion("Back of the pendant", open=True):
                        p_back = gr.Radio(pendant.BACKS, value=pendant.BACKS[0], label="Back")
                        p_font = gr.Radio(list(engrave.FONTS), value=list(engrave.FONTS)[1], label="Font")
                        p_line1 = gr.Textbox(label="Line 1", placeholder="Ram ♥ Meera", max_lines=1,
                                             max_length=20, info="♥ is engraved as a small heart")
                        p_line2 = gr.Textbox(label="Line 2", placeholder="Forever", max_lines=1, max_length=20)
                        p_date = gr.Textbox(label="Date", placeholder="12.02.2015", max_lines=1, max_length=14)
                    with gr.Accordion("Adjust (only if the automatic result looks wrong)", open=False):
                        p_turn = gr.Radio(royal.TURNS, value="auto", label="Turn head (face direction)")
                        p_pupils = gr.Checkbox(value=True, label="Pupil dots (tiny dimples - eyes look alive)")
                    with gr.Accordion("Customer", open=True):
                        p_cname = gr.Textbox(label="Customer name")
                        p_cphone = gr.Textbox(label="Phone / WhatsApp")
                        p_consent = gr.Checkbox(label="Customer agrees we use this photo/model only for "
                                                "their order (and, for the API, that it is sent to Tripo)")
                    p_go = gr.Button("Make pendant", variant="primary")
                with gr.Column(scale=1):
                    p_render = gr.Gallery(label="Front | back", height=440, columns=1)
                    p_3d = gr.Model3D(label="3D preview", height=320)
                    p_files = gr.File(label="Downloads (STL + jeweller notes for the casting shop)",
                                      file_count="multiple")
                    p_quote = gr.Markdown()
                    p_log = gr.Markdown()
            p_go.click(run_pendant,
                       [p_source, p_model1, p_photo1, p_label1, p_pair, p_model2, p_photo2, p_label2, p_shape,
                        p_size, p_crop, p_rim, p_detail, p_back, p_metal, p_font, p_line1, p_line2, p_date, p_pupils,
                        p_turn, p_pack, p_cname, p_cphone, p_consent],
                       [p_render, p_3d, p_files, p_quote, p_log])

        with gr.Tab("📋 Orders"):
            o_table = gr.Dataframe(headers=ORDER_COLS, value=order_table, interactive=False,
                                   wrap=True)
            with gr.Row():
                o_id = gr.Textbox(label="Order ID")
                o_status = gr.Dropdown(STATUSES, value="approved", label="New status")
                o_set = gr.Button("Update status")
                o_refresh = gr.Button("Refresh")
            o_msg = gr.Markdown()
            o_set.click(set_status, [o_id, o_status], [o_table, o_msg])
            o_refresh.click(order_table, None, o_table)

        with gr.Tab("⚙️ Settings"):
            s_status = gr.Markdown(engine_status())
            with gr.Row():
                s_key = gr.Textbox(label="Meshy API key (kept in memory only)", type="password",
                                   value=_session["meshy_key"])
                s_save_key = gr.Button("Use key")
            s_save_key.click(save_key, s_key, s_status)
            with gr.Row():
                s_tkey = gr.Textbox(label="Tripo API key (kept in memory only; or put TRIPO_API_KEY=... "
                                          "in the .env file)", type="password", value=_session["tripo_key"])
                s_save_tkey = gr.Button("Use Tripo key")
            s_save_tkey.click(save_tripo_key, s_tkey, s_status)
            gr.Markdown("### Prices & material costs (`config/pricing.json`)")
            s_prices = gr.Code(value=load_prices(), language="json", lines=30)
            s_save = gr.Button("Save prices")
            s_msg = gr.Markdown()
            s_save.click(save_prices, s_prices, s_msg)
    return ui


def main():
    ap = argparse.ArgumentParser(description="Memory Factory Studio")
    ap.add_argument("--lan", action="store_true", help="listen on the local network")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    build_ui().queue().launch(
        server_name="0.0.0.0" if args.lan else "127.0.0.1",
        server_port=args.port,
        inbrowser=not args.no_browser,
        allowed_paths=[str(orders.ORDERS_DIR)],
    )


if __name__ == "__main__":
    main()
