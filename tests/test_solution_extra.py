"""Phase 2 of the response solution: next passes, warnings, clean-up progress, case file."""
import sys
import types
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services import solution_extra as X  # noqa: E402
from app.services.response_plan import plan  # noqa: E402
from app.services.solution import build  # noqa: E402

LAT, LON = 33.566, -117.915


def incident(log=None):
    site = {"name": "Newport Beach", "type": "public beach", "lat": 33.6, "lon": -117.9, "sensitivity": 4, "eta_h": 0}
    return types.SimpleNamespace(
        id=2, centroid_lat=LAT, centroid_lon=LON, area_km2=4.4, confidence=0.9, status="confirmed", priority="Critical",
        detected_at=datetime(2021, 10, 3, 1, 49), scene_id=1, data={
            "spill": {"area_km2": 4.4, "wind_speed_ms": 4.7, "reliability": "high", "confidence": 0.9},
            "impact": {"volume": {"m3": [1.3, 220]}, "first_landfall_h": 0, "coastline_affected_km": 26.25, "threatened_sites": [site]},
            "forecast": [{"hours": h, "centroid_lat": LAT + 0.001 * h, "centroid_lon": LON - 0.001 * h} for h in (6, 12, 24, 48, 72)],
            "evidence": {"files": [{"role": "satellite image", "name": "S1B.zip", "sha256": "abc"}]},
            "cleanup_log": log or []})


def fake_search(aoi, start, end, cog_only=True):
    times = [("S1B", datetime(2021, 9, 21, 1, 49)), ("S1B", datetime(2021, 10, 3, 1, 49)), ("S1A", datetime(2021, 9, 27, 13, 51)),
             ("S1A", datetime(2021, 10, 9, 13, 51))]
    return [{"Name": f"{s}_IW_GRDH_x", "ContentDate": {"Start": t.isoformat() + ".000000Z"}} for s, t in times]


def test_next_passes_follow_the_12_day_repeat():
    f = X.followup(incident(), search=fake_search)
    assert [p["time"] for p in f["predicted"]] == ["2021-10-09T13:51:00Z", "2021-10-15T01:49:00Z"]
    assert f["predicted"][0]["in_catalogue"]
    assert f["images_after"][0]["hours_after"] == 156.0


def test_warnings_in_three_languages_name_the_threatened_beach():
    inc = incident()
    w = X.warnings(inc, build(inc, []))
    assert all("Newport Beach" in w[k] for k in ("en", "hi", "or"))
    assert "तेल" in w["hi"] and "ତେଲ" in w["or"]


def test_cleanup_progress_against_response_estimate():
    inc = incident([{"oil_m3": 10, "shore_km": 2, "cost": 50000}])
    c = X.cleanup(inc, plan(inc))
    assert c["totals"] == {"oil_m3": 10, "shore_km": 2, "cost": 50000}
    assert 0 < c["progress"]["oil"] < 100


def test_case_file_contains_suspect_and_samples():
    inc = incident()
    s = types.SimpleNamespace(rank=1, vessel_name="KITE", mmsi=338310229, vessel_type="Pleasure craft", score=0.39,
                              review_status="confirmed", evidence=["Passed 0.1 km from the spill"])
    html = X.casefile_html(inc, types.SimpleNamespace(filename="S1B.zip"), [s], build(inc, []), plan(inc), "admin")
    assert "KITE" in html and "Oil Record Book" in html and "polluter pays" in html
