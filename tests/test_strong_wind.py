"""Strong-wind reliability (Kerala monsoon test, 28 May 2025): above ~10 m/s dark patches are mostly rain cells
or wind shadows, so the look-alike check must lean towards look-alike, not towards oil.
    python -m pytest tests/test_strong_wind.py -q
"""
import numpy as np

from ml import config
from ml.lookalike.second_stage import check


def _patch():
    img = np.full((60, 60), 160, np.uint8)
    mask = np.zeros((60, 60), bool)
    mask[25:35, 5:55] = True
    img[mask] = 90
    return img, mask


def _wind_effect(ws):
    img, mask = _patch()
    res = check(img, mask, {"elongation": 4.3, "area_km2": 35.0, "confidence": 0.8}, ws)
    return next(e["effect"] for e in res["evidence"] if e["name"] == "Wind"), res["oil_score"]


def test_strong_wind_counts_against_oil():
    good, s_good = _wind_effect(6.0)
    strong, s_strong = _wind_effect(11.0)
    very, s_very = _wind_effect(12.4)
    assert good > 0 and strong < 0 and very < strong
    assert s_very < s_good


def test_reliable_wind_range():
    assert config.MIN_RELIABLE_WIND_MS == 3.0 and config.MAX_RELIABLE_WIND_MS == 10.0


def test_huge_dark_area_rejected_where_wind_is_strong():
    from ml.pipeline import reject_calm_areas
    cm = np.zeros((400, 400), np.uint8)
    cm[10:200, 10:200] = config.OIL_CLASS          # ~440 km2 dark area
    cm[300:310, 300:330] = config.OIL_CLASS        # small slick
    t = (0.001, 0, 75.7, 0, -0.001, 9.9)
    out, n = reject_calm_areas(cm.copy(), t, 8.0, wind_at=lambda la, lo: 12.4)   # scene centre 8, local 12.4
    assert n == 1 and (out[10:200, 10:200] == config.LOOKALIKE_CLASS).all()
    assert (out[300:310, 300:330] == config.OIL_CLASS).all()                     # small slicks are kept
    _, n = reject_calm_areas(cm.copy(), t, 8.0, wind_at=lambda la, lo: 6.0)
    assert n == 0                                                                # normal wind: big slick kept
