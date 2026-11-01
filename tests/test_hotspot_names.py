"""Hotspot names: nearest site or coastal town, never a place on the other side of the world."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.analytics.hotspots import hotspots, nearest_place  # noqa: E402

CALIFORNIA_SITE = [{"name": "Laguna Beach", "lat": 33.54, "lon": -117.78}]


def test_kerala_hotspot_named_after_indian_town():
    name = nearest_place(9.4, 75.9, CALIFORNIA_SITE)
    assert "Laguna" not in name and "km" in name
    assert name.startswith("Alappuzha") or name.startswith("Kochi")


def test_california_hotspot_keeps_site_name():
    assert nearest_place(33.55, -117.80, CALIFORNIA_SITE).startswith("Laguna Beach")


def test_far_offshore_is_open_sea():
    assert nearest_place(5.0, 65.0).startswith("Open sea")


def test_hotspots_use_new_names():
    h = hotspots([{"id": 1, "lat": 9.4, "lon": 75.9, "area_km2": 2, "priority": "Low", "status": "detected"}], CALIFORNIA_SITE)
    assert "Laguna" not in h[0]["near"]
