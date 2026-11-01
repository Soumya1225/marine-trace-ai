"""Hotspot grid and trend: repeated spills in one cell rank first; rejected incidents are ignored."""
from ml.analytics.hotspots import hotspots, trend

INC = [
    {"id": 1, "lat": 33.61, "lon": -118.05, "area_km2": 4.4, "priority": "Critical", "status": "detected", "date": "2021-10-03", "verdict": "Likely oil"},
    {"id": 2, "lat": 33.62, "lon": -118.04, "area_km2": 1.0, "priority": "High", "status": "detected", "date": "2021-10-03", "verdict": "Likely oil"},
    {"id": 3, "lat": 33.40, "lon": -118.40, "area_km2": 0.2, "priority": "Low", "status": "detected", "date": "2021-10-04", "verdict": "Likely look-alike"},
    {"id": 4, "lat": 33.40, "lon": -118.40, "area_km2": 9.0, "priority": "Critical", "status": "rejected", "date": "2021-10-04"},
]
SITES = [{"name": "Huntington Beach", "lat": 33.65, "lon": -118.0}]


def test_hotspot_ranking():
    h = hotspots(INC, SITES)
    assert h[0]["incident_ids"] == [1, 2] and h[0]["count"] == 2
    assert "Huntington Beach" in h[0]["near"]
    assert all(4 not in x["incident_ids"] for x in h)


def test_trend_split_by_verdict():
    t = {d["date"]: d for d in trend(INC)}
    assert t["2021-10-03"]["oil"] == 2 and t["2021-10-04"]["lookalike"] == 1 and t["2021-10-04"]["count"] == 1
