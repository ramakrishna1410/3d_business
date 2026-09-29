"""Run with:  python -m pytest -q"""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from memory_factory import config, drawing, medallion, mesh


@pytest.fixture(autouse=True)
def tmp_orders(tmp_path, monkeypatch):
    import memory_factory.orders as orders

    monkeypatch.setattr(orders, "ORDERS_DIR", tmp_path / "orders")
    monkeypatch.setattr(config, "ORDERS_DIR", tmp_path / "orders")


def couple_photo():
    im = Image.new("RGB", (400, 520), (90, 140, 200))
    d = ImageDraw.Draw(im)
    for cx in (130, 270):
        d.ellipse([cx - 55, 90, cx + 55, 230], fill=(220, 180, 150))
        d.rectangle([cx - 80, 240, cx + 80, 520], fill=(180, 40, 60))
    return im


def kid_drawing():
    im = Image.new("RGB", (600, 450), (250, 248, 240))
    d = ImageDraw.Draw(im)
    d.ellipse([150, 120, 430, 330], fill=(80, 170, 90), outline=(20, 20, 20), width=5)
    d.ellipse([380, 60, 520, 190], fill=(80, 170, 90), outline=(20, 20, 20), width=5)
    d.rectangle([200, 310, 230, 420], fill=(80, 170, 90), outline=(20, 20, 20), width=5)
    d.rectangle([340, 310, 370, 420], fill=(80, 170, 90), outline=(20, 20, 20), width=5)
    return im


def test_heightmap_mesh_is_closed_with_holes_and_pinches():
    h = np.ones((40, 40))
    m = np.ones((40, 40), bool)
    m[10:20, 10:20] = False          # a hole
    m[30, 30] = m[31, 31] = False    # diagonal pinch pattern
    solid = mesh.heightmap_to_mesh(h + 1, m, 0.5)
    assert mesh.is_watertight(solid)
    assert solid.volume_mm3() > 0


def test_volume_of_flat_slab():
    solid = mesh.heightmap_to_mesh(np.full((11, 21), 2.0), None, 1.0)
    assert solid.volume_mm3() == pytest.approx(10 * 20 * 2.0)


def test_medallion_end_to_end():
    s = medallion.MedallionSettings(diameter_mm=40, pixel_mm=0.2, keychain_hole=True,
                                    remove_background=False, depth_engine="fast")
    r = medallion.generate(couple_photo(), s, {"name": "Test"}, quantity=120)
    assert r.stl.exists() and r.proof.exists()
    assert 40 * 40 * 2.0 < r.volume_mm3 < 40 * 40 * 5.0
    assert "Suggested price" in r.quote_md
    assert (r.folder / "order.json").exists()


@pytest.mark.parametrize("mode", ["relief plaque", "standing figure"])
def test_drawing_modes(mode):
    s = drawing.DrawingSettings(mode=mode, size_mm=60, pixel_mm=0.4)
    r = drawing.generate(kid_drawing(), s, {"name": "Parent"})
    assert r.stl.exists() and r.painting_guide.exists()
    assert r.volume_mm3 > 0
    assert any("watertight shells=True" in line for line in r.log)


def test_import_model(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    trimesh.creation.icosphere().export(tmp_path / "blob.glb")
    s = drawing.DrawingSettings(mode="import 3d model", size_mm=50, pixel_mm=0.4)
    r = drawing.generate(kid_drawing(), s, {}, model_file=tmp_path / "blob.glb")
    assert r.stl.exists()
    assert 50 < r.height_mm < 50 + 12  # figure + name base


def test_blank_page_is_rejected():
    with pytest.raises(ValueError):
        drawing.generate(Image.new("RGB", (300, 300), "white"), drawing.DrawingSettings(), {})
