"""Response solution: tier, clean-up method ranking, intercept point and checklist."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.solution import build  # noqa: E402


def make(lat, lon, area, wind, reliability="high", sites=None, landfall=None):
    fc = [{"hours": h, "centroid_lat": lat + 0.004 * h / 6, "centroid_lon": lon - 0.003 * h / 6} for h in (6, 12, 24, 48, 72)]
    return types.SimpleNamespace(centroid_lat=lat, centroid_lon=lon, area_km2=area, data={
        "spill": {"area_km2": area, "wind_speed_ms": wind, "reliability": reliability, "confidence": 0.9,
                  "polygon": [[lon, lat], [lon + 0.01, lat], [lon, lat + 0.01]]},
        "impact": {"volume": {"m3": [area * 0.3, area * 50]}, "first_landfall_h": landfall,
                   "coastline_affected_km": 2.0, "threatened_sites": sites or []},
        "forecast": fc})


def test_calm_sea_near_coast_recommends_booms_and_shoreline():
    site = {"name": "Wetland", "type": "wetland", "lat": 33.70, "lon": -118.05, "sensitivity": 5, "eta_h": 30}
    r = build(make(33.566, -117.915, 4.4, 3.0, sites=[site], landfall=36), [{"vessel_name": "KITE", "mmsi": 1, "score": 0.7}])
    assert r["method"]["recommended"] == "mechanical"
    assert "shoreline" in r["method"]["summary"].lower()
    assert r["tier"]["tier"] == 1
    assert r["intercept"]["reached"]
    assert any(t["id"] == "investigate" for t in r["checklist"]["tasks"])


def test_low_reliability_asks_to_verify_first():
    r = build(make(9.5, 75.8, 0.5, 2.0, reliability="low"), [])
    assert r["method"]["verify_first"]
    assert r["method"]["recommended"] == "monitor"
    assert r["checklist"]["tasks"][0]["id"] == "verify"


def test_big_offshore_spill_in_india_is_tier_2_with_coast_guard_lead():
    r = build(make(15.0, 72.5, 40.0, 8.0), [])
    assert r["tier"]["tier"] == 2
    assert "Indian Coast Guard" in r["tier"]["lead_agency"]
    assert r["method"]["recommended"] == "dispersant"
