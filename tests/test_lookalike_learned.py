"""B4: trained look-alike model. A tiny made-up dataset (sharp thin streaks = oil, large soft blobs = look-alikes)
stands in for the real one, so this runs in seconds:
    python -m pytest tests/test_lookalike_learned.py -q
"""
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_lookalike_med", ROOT / "scripts" / "train_lookalike_med.py")
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)
from ml.lookalike import learned  # noqa: E402
from ml.lookalike.second_stage import check  # noqa: E402

XML = ("<annotation><object><name>oil</name><bndbox><xmin>{}</xmin><ymin>{}</ymin><xmax>{}</xmax><ymax>{}</ymax>"
       "</bndbox></object></annotation>")


def _sea(rng, n=320):
    return np.clip(rng.normal(150, 18, (n, n)), 1, 255)


def _make(tmp, rng, k=24):
    for s in ("ow", "nw"):
        (tmp / s).mkdir(parents=True)
    for i in range(k):
        img = _sea(rng)
        r, c0 = int(rng.integers(60, 260)), int(rng.integers(20, 80))
        img[r:r + 8, c0:c0 + 200] = rng.normal(60, 6, (8, 200))            # thin, sharp, very dark
        Image.fromarray(img.astype(np.uint8)).save(tmp / "ow" / f"ow-{i:04d}.jpg")
        (tmp / "ow" / f"ow-{i:04d}.xml").write_text(XML.format(c0 - 5, r - 5, c0 + 205, r + 13))
        img = _sea(rng)
        yy, xx = np.mgrid[:320, :320]
        cy, cx = rng.integers(100, 220, 2)
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 55.0 ** 2))
        img = img - 70 * ndimage.gaussian_filter(blob, 4)                    # big, soft, mildly dark
        Image.fromarray(np.clip(img, 1, 255).astype(np.uint8)).save(tmp / "nw" / f"nw-{i:04d}.jpg")


def test_train_and_use_learned_model(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    _make(tmp_path, rng)
    monkeypatch.setattr(tr, "TAB", tmp_path / "none.tab")
    X, y, groups, rule = tr.build(tmp_path)
    assert X.shape[1] == len(learned.FEATURES) and y.sum() >= 20 and (1 - y).sum() >= 20
    res, best, model, _, _ = tr.evaluate(X, y, groups, rule)
    assert res[best]["auc"] > 0.9
    path = tmp_path / "m.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"model": model, "features": learned.FEATURES}, fh)
    monkeypatch.setitem(learned._MODEL, "loaded", False)
    learned.load(path)
    img = tr.load_img(tmp_path / "ow" / "ow-0000.jpg")
    m = tr.oil_samples(img, tmp_path / "ow" / "ow-0000.xml")[0]
    ci, cm = tr.crop(img, m)
    monkeypatch.setattr(sys.modules["ml.lookalike.second_stage"], "LEARNED_WEIGHT", 0.5)
    out = check(ci, cm, {"elongation": 8.0})
    assert out["learned_features"] and out["learned_outside"] == []
    assert out["learned_oil_prob"] is not None and out["learned_oil_prob"] > 0.5
    assert any(e["name"] == "Trained look-alike model" for e in out["evidence"])
    monkeypatch.setitem(learned._MODEL, "obj", None)                       # no model -> rules only
    assert check(ci, cm, {"elongation": 8.0})["learned_oil_prob"] is None


def test_resample_to_model_pixel_size():
    img = np.full((200, 200), 150.0)
    mask = np.zeros((200, 200), bool)
    mask[90:110, 20:180] = True
    img2, m2 = learned.to_target(img, mask, pixel_m=20.0)                 # 20 m -> 80 m
    assert img2.shape == (50, 50) and 120 < m2.sum() < 260
    assert learned.to_target(img, mask, pixel_m=85.0)[0].shape == (200, 200)   # already close: unchanged
    pm = learned.pixel_size_m((0.0002, 0, -118.2, 0, -0.0002, 33.7), 33.5)
    assert 18 < pm < 22


def test_learned_model_is_display_only_by_default():
    from ml.lookalike import second_stage
    img = np.full((60, 60), 160.0)
    mask = np.zeros((60, 60), bool)
    mask[25:35, 5:55] = True
    img[mask] = 90
    lr = {"p": 0.0, "features": {}, "outside": []}
    out = check(img, mask, {"elongation": 4.0}, 6.0, use_learned=False, learned_result=lr)
    assert second_stage.LEARNED_WEIGHT == 0 and out["oil_score"] == out["rule_score"]
    assert out["learned_oil_prob"] == 0.0
