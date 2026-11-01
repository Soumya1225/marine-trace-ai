"""Tests for the Oct 2026 features: assistant, response planning, summary page, security, watch zones, live AIS.

They use small fake incidents, so they run in a second without models, data or a database:
    python -m pytest tests/test_new_features.py -q
"""
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace as N

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.assistant import _intent, answer  # noqa: E402
from app.services.response_plan import plan  # noqa: E402
from app.services.summary_page import build  # noqa: E402


def make_incident(lat=33.566, lon=-117.915):
    shap = {"speed_ratio": 0.4, "loiter_h": 0.2, "alignment": -0.3}
    sus = [N(rank=i + 1, vessel_name=n, mmsi=338000000 + i, vessel_type="Tanker", score=0.6 - i * 0.1, rule_score=0.7,
             ml_score=0.5, evidence=["Passed 0.1 km from the spill."], review_status="pending", features={"shap": shap})
           for i, n in enumerate(["KITE", "ALLURE", "LANI KAI"])]
    return N(id=2, title="Spill 2", priority="Critical", priority_score=88, status="detected", area_km2=4.4, confidence=0.9,
             centroid_lat=lat, centroid_lon=lon, detected_at=datetime(2021, 10, 3, 1, 49), suspects=sus,
             assignee=None, due_at=None, closed_at=None,
             data={"spill": {"wind_speed_ms": 4.7, "reliability": "normal",
                             "polygon": [[lon, lat], [lon + 0.03, lat], [lon + 0.03, lat + 0.03], [lon, lat + 0.03]],
                             "lookalike_check": {"verdict": "Likely oil", "oil_score": 0.7, "evidence": [{"text": "Wind ok."}]}},
                   "forecast": [{"hours": 6, "area_km2": 6.0, "centroid": [lon, lat]}, {"hours": 24, "area_km2": 12.0, "centroid": [lon, lat]}],
                   "impact": {"threatened_sites": [{"name": "Newport Beach", "type": "beach", "eta_h": None}],
                              "first_landfall_h": 12, "coastline_affected_km": 8.5, "volume": {"m3": [1, 200], "barrels": [6, 1258]}},
                   "decision": {"priority": "Critical", "priority_score": 88, "actions": [{"action": "Deploy booms"}]},
                   "ships": {"targets": [{"dark": True, "lat": lat, "lon": lon}]}})


@pytest.mark.parametrize("q,intent", [
    ("Which ship is most suspicious and why?", "why"), ("Where will the oil be in 6 hours?", "forecast"),
    ("Which beaches are threatened?", "impact"), ("Is this really oil or a look-alike?", "reliability"),
    ("What should we do now?", "actions"), ("Were there any ships without AIS?", "dark"), ("who is assigned", "case"),
])
def test_assistant_intents(q, intent):
    assert _intent(q) == intent


def test_assistant_answers_from_data():
    inc = make_incident()
    assert "KITE" in answer(inc, "Which ship is most suspicious and why?")["answer"]
    assert "ALLURE" in answer(inc, "Why is ALLURE suspected?")["answer"]
    assert "Newport Beach" in answer(inc, "Which beaches are threatened?")["answer"]
    assert "6 h" in answer(inc, "Where will the oil be in 6 hours?")["answer"]


def test_response_plan_nearest_base_and_costs():
    r = plan(make_incident(20.2, 86.8))                       # off Paradip, Odisha
    assert r["nearest"][0]["name"] == "ICG Station Paradip"
    assert all(s["distance_km"] <= 1000 for s in r["nearest"])
    assert r["equipment"]["boom_m"] > 0
    assert r["cost"]["total"][0] <= r["cost"]["total"][1]
    assert r["timing"]["arrives_before_landfall"] is True


def test_summary_page_hindi():
    html = build(make_incident(), "hi")
    assert 'lang="hi"' in html and "तेल रिसाव" in html and "KITE" in html


def test_password_policy():
    from fastapi import HTTPException
    from app.security import check_password
    check_password("Spill2026")
    for bad in ("short1", "longpassword", "12345678"):
        with pytest.raises(HTTPException):
            check_password(bad)


def test_login_rate_limit():
    from fastapi import HTTPException
    from app.security import MAX_FAILS, login_allowed, login_failed, login_ok
    for _ in range(MAX_FAILS):
        login_allowed("x@test", "1.2.3.4")
        login_failed("x@test", "1.2.3.4")
    with pytest.raises(HTTPException):
        login_allowed("x@test", "1.2.3.4")
    login_ok("x@test", "1.2.3.4")
    login_allowed("x@test", "1.2.3.4")


def test_zone_contains_point():
    from app.routers.zones import inside
    z = N(min_lon=86.5, min_lat=20.1, max_lon=86.9, max_lat=20.4)
    assert inside(z, 20.2, 86.7) and not inside(z, 19.0, 86.7)


def test_live_ais_message_parsing():
    from app.services.ais_live import LiveAIS
    live = LiveAIS()
    live._handle({"MessageType": "PositionReport", "MetaData": {"MMSI": 419000001, "ShipName": "TEST TANKER"},
                  "Message": {"PositionReport": {"Latitude": 20.2, "Longitude": 86.7, "Sog": 10.5, "Cog": 90, "TrueHeading": 511}}})
    live._handle({"MessageType": "ShipStaticData", "MetaData": {"MMSI": 419000001},
                  "Message": {"ShipStaticData": {"Type": 80, "Dimension": {"A": 150, "B": 30}}}})
    s = live.snapshot()[0]
    assert s["name"] == "TEST TANKER" and s["type"] == 80 and s["length"] == 180 and s["heading"] is None
