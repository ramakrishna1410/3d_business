"""Run with:  python -m pytest -q"""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from memory_factory import config, drawing, imaging, medallion, mesh


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
    solid = mesh.heightmap_to_mesh(np.full((11, 21), 2.0), None, 1.0, smooth_edges=False)
    assert solid.volume_mm3() == pytest.approx(10 * 20 * 2.0)


def test_smoothed_disc_edge_is_round_and_closed():
    n, r = 201, 90
    yy, xx = np.mgrid[0:n, 0:n]
    disc = np.hypot(xx - 100, yy - 100) <= r
    rough = mesh.heightmap_to_mesh(np.full((n, n), 2.0), disc, 1.0, smooth_edges=False)
    smooth = mesh.heightmap_to_mesh(np.full((n, n), 2.0), disc, 1.0)
    assert mesh.is_watertight(smooth)

    def edge_error(m):  # wobble of the silhouette radius around the circle
        v = m.vertices
        rr = np.hypot(v[:, 0] - 100, v[:, 1] - 100)
        ang = ((np.arctan2(v[:, 1] - 100, v[:, 0] - 100) + np.pi) / (2 * np.pi) * 360).astype(int)
        outer = np.full(361, -1.0)
        np.maximum.at(outer, ang, rr)
        return np.std(outer[outer > 0])

    assert edge_error(smooth) < 0.6 * edge_error(rough)   # staircase flattened
    assert smooth.volume_mm3() == pytest.approx(rough.volume_mm3(), rel=0.01)


@pytest.mark.parametrize("layout,hole", [("coin", False), ("coin", True), ("classic", True)])
def test_medallion_end_to_end(layout, hole):
    s = medallion.MedallionSettings(diameter_mm=40, pixel_mm=0.2, keychain_hole=hole,
                                    layout=layout, remove_background=False, depth_engine="fast")
    r = medallion.generate(couple_photo(), s, {"name": "Test"}, quantity=120)
    assert any("watertight=True" in line for line in r.log)
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


def test_arc_text_bottom_and_top():
    n, c = 301, 150
    bottom = medallion._arc_text(n, c, "RAM & MEERA", 120, 20, "bottom", None)
    top = medallion._arc_text(n, c, "22.04.2026", 120, 16, "top", None)
    rows_b = np.nonzero(bottom.max(axis=1) > 0.5)[0]
    rows_t = np.nonzero(top.max(axis=1) > 0.5)[0]
    assert rows_b.min() > c and rows_t.max() < c   # bottom arc below centre, top arc above
    assert bottom.sum() > 0 and top.sum() > 0


def test_portrait_crop_keeps_faces_and_aspect():
    mask = np.ones((800, 600))
    faces = [(150, 200, 120, 120), (330, 230, 110, 110)]
    x0, y0, x1, y1 = imaging.portrait_crop_box((600, 800), mask, faces, aspect=1.1)
    assert x0 <= 150 and x1 >= 440 and y0 <= 200 and y1 >= 340
    assert abs((x1 - x0) / (y1 - y0) - 1.1) < 0.1


def leaf_sketch():
    """Pencil-style outline: a big closed shape with a figure drawn inside it."""
    im = Image.new("RGB", (500, 600), (235, 235, 230))
    d = ImageDraw.Draw(im)
    d.ellipse([40, 40, 460, 560], outline=(60, 60, 60), width=6)       # leaf outline
    d.line([250, 45, 250, 555], fill=(90, 90, 90), width=3)            # vein
    d.ellipse([170, 170, 330, 330], outline=(50, 50, 50), width=4)     # head (figure)
    d.rectangle([180, 340, 320, 470], outline=(50, 50, 50), width=4)   # body (figure)
    return im


def test_lineart_layers_and_mesh():
    from memory_factory import lineart

    s = lineart.LineArtSettings(size_mm=80, pixel_mm=0.3)
    log = []
    height, solid, _ = lineart.build(leaf_sketch(), s, log)
    assert "figure" in log[0]
    h = height[solid]
    # figure layer stands above the background layer, which stands above the base
    assert h.max() > s.base_mm + s.figure_mm
    assert np.percentile(h, 20) >= s.base_mm
    m = mesh.heightmap_to_mesh(height, solid, s.pixel_mm)
    assert mesh.is_watertight(m)


def test_drawing_lineart_mode():
    s = drawing.DrawingSettings(mode="line-art relief", size_mm=80)
    r = drawing.generate(leaf_sketch(), s, {"name": "Parent"})
    assert r.stl.exists() and r.render.exists()
    assert any("watertight shells=True" in line for line in r.log)


def test_lineart_result_does_not_depend_on_print_size():
    from memory_factory import lineart

    counts = []
    for size in (50, 150):
        log = []
        lineart.build(leaf_sketch(), lineart.LineArtSettings(size_mm=size, pixel_mm=0.3), log)
        counts.append(log[0])
    assert counts[0] == counts[1]


@pytest.mark.parametrize("kind", ["king", "queen", "bishop", "knight", "rook", "pawn"])
def test_chess_pieces_are_closed_and_sized(kind):
    from memory_factory import chess

    m = chess.piece(kind, None, "A")
    lo, hi = m.bounds()
    assert abs((hi[2] - lo[2]) - chess.SPECS[kind].height) < 12
    assert lo[2] >= -0.01          # stands on the table
    assert m.volume_mm3() > 0


def test_chess_photo_window_and_sheet():
    from memory_factory import chess

    img = Image.new("RGB", (160, 200), (200, 150, 120))
    face = chess.Face(img, np.ones((200, 160)), np.zeros((200, 160)), (40, 40, 80, 80))
    m = chess.piece("king", face, style="photo")
    assert m.volume_mm3() > 0
    photo = chess.window_photo(face, *chess.window_size_mm("king"))
    w_mm, h_mm = chess.window_size_mm("king")
    assert abs(photo.width - w_mm / 25.4 * 300) < 2       # printed at true size
    cols = chess.photo_face_colors(m, "king", photo, (214, 168, 82))
    assert (cols != [214, 168, 82]).any(axis=1).sum() > 100  # photo shows in the window
    sheet = chess.photo_sheet([("King", photo)])
    assert sheet.size == (2480, 3507)


# ---------------------------------------------------------------- Royal Chess
def synthetic_bust(tmp_path, turn_deg=0):
    """Head (with nose, lips, chin) on a neck and shoulders, face towards -y, z-up."""
    trimesh = pytest.importorskip("trimesh")
    parts = [
        trimesh.creation.box((44, 18, 12), trimesh.transformations.translation_matrix((0, 0, 6))),
        trimesh.creation.cylinder(4.5, 14, transform=trimesh.transformations.translation_matrix((0, 0, 17))),
        trimesh.creation.icosphere(4, 11.0).apply_scale((0.85, 1.0, 1.15)).apply_translation((0, 0, 33)),
        trimesh.creation.icosphere(3, 2.4).apply_translation((0, -10.6, 32)),     # nose
        trimesh.creation.icosphere(3, 1.6).apply_translation((0, -9.6, 27.5)),    # lips
        trimesh.creation.icosphere(3, 2.6).apply_translation((0, -7.5, 23.5)),    # chin
    ]
    m = trimesh.util.concatenate(parts)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(turn_deg), [0, 0, 1]))
    path = tmp_path / f"head_{turn_deg}.stl"
    m.export(path)
    return path


@pytest.mark.parametrize("turn", [0, 90, 180])
def test_royal_finds_the_face(tmp_path, turn):
    from memory_factory import royal

    path = synthetic_bust(tmp_path, turn)
    v = royal.orient(royal.load_head(path), ".stl", "auto", [])
    top = v[v[:, 2] > 20]
    nose = top[np.argmin(top[:, 1])]
    assert nose[1] < -9 and abs(nose[0]) < 3      # nose ends up pointing to -y


@pytest.mark.parametrize("style", ["smooth statue", "classic royal"])
@pytest.mark.parametrize("piece", ["king", "queen", "bishop"])
def test_royal_piece_is_one_watertight_solid(tmp_path, piece, style):
    pytest.importorskip("manifold3d")
    pytest.importorskip("skimage")
    from memory_factory import royal

    s = royal.RoyalSettings(piece=piece, style=style, size=list(royal.SIZES)[0], quality="preview")
    log = []
    p, _ = royal.build_piece(synthetic_bust(tmp_path), s, log)
    assert p.is_watertight
    assert len(p.split(only_watertight=False)) == 1
    assert 60 < p.bounds[1][2] < 95                # chess-set size (King ~80 mm)
    assert abs(p.bounds[0][2]) < 0.5               # stands on the table


def test_tidy_hair_removes_loose_strands_but_keeps_the_face():
    from memory_factory import royal

    g = royal.Grid((-16, -16, royal.Z_NECK - 1), (16, 16, royal.Z_NECK + 26), 0.25)
    X, Y, Z = g.xyz()
    zc = royal.Z_NECK + 12
    head = np.hypot(np.hypot(X / 8, Y / 9), (Z - zc) / 12) <= 1
    nose = np.hypot(np.hypot(X, Y + 9.6), Z - zc) <= 1.2
    strand = (np.abs(X - 11) < 0.2) & (np.abs(Y - 2) < 0.2) & (Z < zc)        # 0.4 mm hair strand
    hair = (np.hypot(X, Y - 6) < 6) & (Z < royal.Z_NECK + 3)                  # long hair behind the neck
    ear_bridge = (np.abs(Y - 2) < 0.2) & (np.abs(X) < 11.2) & (np.abs(Z - zc) < 0.2)
    solid = head | nose | strand | hair | ear_bridge
    out = royal.tidy_hair(solid.copy(), g, [])
    assert not out[(X > 10.5) & (Z < zc - 3) & np.broadcast_to(True, out.shape)].any()   # strand gone
    assert out[nose & np.broadcast_to(True, out.shape)].mean() > 0.95                  # face kept
    sl = out[:, :, np.searchsorted(g.z, royal.Z_NECK + 0.5)]
    assert g.y[np.nonzero(sl)[1]].max() < royal.COLLAR[1] + 1.0       # hair ends inside the collar


@pytest.mark.parametrize("font", ["Script (Great Vibes)", "Royal capitals (Cinzel)", "Elegant italic (Playfair)"])
def test_royal_engraved_name_and_message(tmp_path, font):
    pytest.importorskip("manifold3d")
    from memory_factory import royal

    base = dict(piece="queen", size=list(royal.SIZES)[1], quality="preview")
    plain, _ = royal.build_piece(synthetic_bust(tmp_path), royal.RoyalSettings(**base), [])
    log = []
    s = royal.RoyalSettings(**base, font=font, name="Meera", message1="Ram & Meera", date="05.10.2026")
    p, _ = royal.build_piece(synthetic_bust(tmp_path), s, log)
    assert p.is_watertight and len(p.split(only_watertight=False)) == 1
    assert p.volume < plain.volume - 5                     # letters are cut in, not added
    assert any(l.startswith('Name "Meera"') for l in log) and any("Under the base" in l for l in log)
    notes = royal.print_notes(s, 98)
    assert '"Meera"' in notes and '"Ram & Meera"' in notes and "drain holes" in notes


def test_royal_text_too_long_fails_before_building():
    from memory_factory import royal

    with pytest.raises(ValueError, match="too long"):
        royal.check_text(royal.RoyalSettings(size=list(royal.SIZES)[0], name="Bartholomew Alexander"))
    with pytest.raises(ValueError, match="too long"):
        royal.check_text(royal.RoyalSettings(size=list(royal.SIZES)[0],
                                             message1="With love from all of us at home, forever"))
    royal.check_text(royal.RoyalSettings(size=list(royal.SIZES)[1], name="Lakshmi", message1="Ram & Meera",
                                         message2="Forever", date="05.10.2026"))


def test_royal_generate_end_to_end(tmp_path):
    pytest.importorskip("manifold3d")
    from memory_factory import royal

    s = royal.RoyalSettings(piece="king", quality="preview")
    r = royal.generate(synthetic_bust(tmp_path), s, {"name": "Test"})
    assert r.stl.exists() and r.render.exists() and (r.folder / "order.json").exists()
    assert r.extra["print_notes"].exists()
    assert 95 < r.height_mm < 130                  # couple-gift size
    assert "Suggested price" in r.quote_md


class _Resp:
    def __init__(self, data=None, status=200, content=b""):
        self._data, self.status_code, self.content, self.text = data, status, content, ""

    def json(self):
        return self._data


class _FakeTripo:
    """Records calls; plays upload -> task -> running -> success -> download."""

    def __init__(self):
        self.calls, self.polls = [], 0

    def post(self, url, headers=None, files=None, json=None, timeout=None):
        self.calls.append(("POST", url, json))
        assert headers["Authorization"] == "Bearer sk-test"
        if url.endswith("/files"):
            return _Resp({"code": 0, "data": {"file_token": "file_1"}})
        return _Resp({"code": 0, "data": {"task_id": "task_1"}})

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("GET", url, None))
        if url.endswith("/tasks/task_1"):
            self.polls += 1
            if self.polls < 3:
                return _Resp({"code": 0, "data": {"status": "running", "progress": 40}})
            return _Resp({"code": 0, "data": {"status": "success", "credits_consumed": 30,
                                              "output": {"model_url": "https://cdn/x.glb"}}})
        return _Resp(content=b"glTF-bytes")


def test_tripo_client_flow(tmp_path):
    from memory_factory.providers import tripo

    photo = tmp_path / "p.webp"
    couple_photo().save(photo)
    fake, log = _FakeTripo(), []
    out = tripo.photo_to_head_model(photo, "sk-test", tmp_path, log, session=fake, sleep=lambda s: None)
    assert out.read_bytes() == b"glTF-bytes"
    body = next(c[2] for c in fake.calls if c[1].endswith("/generation/image-to-model"))
    assert body["input"] == "file_1"
    assert body["texture"] is False and body["pbr"] is False      # bare geometry for printing
    assert body["geometry_quality"] == "detailed"
    assert fake.polls == 3 and "30 credits" in log[-1]


def test_tripo_errors_are_readable(tmp_path):
    from memory_factory.providers import tripo

    class Bad(_FakeTripo):
        def post(self, url, **kw):
            return _Resp({"code": 1001, "message": "invalid key"}, status=401)

    with pytest.raises(tripo.TripoError, match="API key rejected"):
        tripo.upload_image(tmp_path / "x.jpg" if (tmp_path / "x.jpg").write_bytes(b"1") else None,
                           "sk-bad", session=Bad())


# ---------------------------------------------------------------- photo check (before Tripo)
def portrait(face=300, level=0.55, sharp=True, size=(900, 1100)):
    """Grey photo with a textured 'face' square at (300, 250)."""
    rng = np.random.default_rng(1)
    a = np.full(size[::-1], 0.6)
    tex = rng.random((face, face)) if sharp else np.tile(np.linspace(0, 1, face), (face, 1))
    a[250:250 + face, 300:300 + face] = level + 0.3 * (tex - 0.5)
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).convert("RGB")


@pytest.fixture
def fake_detector(monkeypatch):
    """Detection is stubbed: the tests check the rules, not OpenCV."""
    from memory_factory import photo_check

    state = {"faces": [(300, 250, 300, 300)]}
    monkeypatch.setattr(imaging, "detect_faces", lambda img, mask=None: state["faces"])
    monkeypatch.setattr(photo_check, "_eyes",
                        lambda gray, f: [(f[2] * 0.3, f[3] * 0.4), (f[2] * 0.7, f[3] * 0.4)])
    return state


def test_photo_check_good_photo_gives_tripo_crop(fake_detector):
    from memory_factory import photo_check

    c = photo_check.check_photo(portrait(), white_background=False)
    assert c.level == "good" and c.ok and c.faces == 1 and c.face_px == 300
    assert c.crop.size == (1024, 1024)
    assert "Photo is good" in c.as_markdown()


def test_photo_check_no_face_is_bad(fake_detector):
    from memory_factory import photo_check

    fake_detector["faces"] = []
    c = photo_check.check_photo(portrait(), white_background=False)
    assert not c.ok and c.crop is None


def test_photo_check_small_face_is_bad(fake_detector):
    from memory_factory import photo_check

    fake_detector["faces"] = [(300, 250, 100, 100)]
    c = photo_check.check_photo(portrait(face=100), white_background=False)
    assert c.level == "bad" and "too small" in " ".join(c.messages)


def test_photo_check_dark_photo_is_bad(fake_detector):
    from memory_factory import photo_check

    c = photo_check.check_photo(portrait(level=0.12), white_background=False)
    assert c.level == "bad" and "dark" in " ".join(c.messages)


def test_photo_check_warnings(fake_detector):
    from memory_factory import photo_check

    fake_detector["faces"] = [(300, 250, 300, 300), (650, 300, 160, 160)]
    c = photo_check.check_photo(portrait(), white_background=False)
    assert c.level == "warning" and c.ok and c.faces == 2
    fake_detector["faces"] = [(300, 250, 300, 300)]
    c = photo_check.check_photo(portrait(sharp=False), white_background=False)
    assert c.level == "warning" and "blurry" in " ".join(c.messages)
