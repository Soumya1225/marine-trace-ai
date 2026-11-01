"""Second-stage look-alike check: a sharp thin slick scores as oil, a diffuse round calm patch as look-alike."""
import numpy as np
from scipy import ndimage

from ml.lookalike.second_stage import check, check_spills


def _sea(seed=0, n=300):
    rng = np.random.default_rng(seed)
    return ndimage.uniform_filter(rng.gamma(4, 30 / 4, (n, n)), 3) + 90


def test_sharp_thin_slick_is_oil():
    img, m = _sea(), np.zeros((300, 300), bool)
    m[140:150, 40:260] = True
    img[m] *= 0.45
    r = check(img, m, {"elongation": 8, "area_km2": 2, "confidence": 0.8}, wind_ms=6.0)
    assert r["verdict"] == "Likely oil" and r["oil_score"] >= 0.6


def test_diffuse_calm_patch_is_lookalike():
    yy, xx = np.mgrid[:300, :300]
    r = np.hypot(yy - 150, xx - 150)
    img = _sea(1) * (1 - 0.25 * np.clip((90 - r) / 60, 0, 1))
    res = check(img, r < 80, {"elongation": 1.1, "area_km2": 40, "confidence": 0.6}, wind_ms=1.0)
    assert res["verdict"] == "Likely look-alike"


def test_check_spills_matches_patch_and_adds_note():
    img, m = _sea(2), np.zeros((300, 300), bool)
    yy, xx = np.mgrid[:300, :300]
    m[np.hypot(yy - 150, xx - 150) < 80] = True
    transform = (0.001, 0, -118.3, 0, -0.001, 33.8)                  # 1 px = 0.001 deg
    spill = {"centroid_lat": 33.8 - 0.150, "centroid_lon": -118.3 + 0.150, "elongation": 1.1, "area_km2": 40,
             "wind_speed_ms": 0.8}
    check_spills(img * (1 - 0.1 * m), m, transform, [spill])
    assert "lookalike_check" in spill
    assert spill["lookalike_check"]["verdict"] == "Likely look-alike"
    assert any(n.startswith("Second-stage") for n in spill["reliability_notes"])
