"""Tests for automatic wind / current data (A5). No network or xarray needed: the downloads are faked.
    python -m pytest tests/test_environment_auto.py -q
"""
from datetime import date, datetime, timezone

import numpy as np
import pytest

from ml import environment_auto as ea

BBOX = (72.70, 18.80, 73.00, 19.10)                      # off Mumbai
WHEN = datetime(2023, 5, 10, 1, 0, tzinfo=timezone.utc)


def fake_openmeteo(kind):
    def fetch(url, params):
        n = len(params["latitude"].split(","))
        hours = [f"2023-05-09T{h:02d}:00" for h in range(3)]
        if kind == "wind":
            assert url == ea.OPENMETEO_WIND and params["wind_speed_unit"] == "ms"
            h = {"time": hours, "wind_speed_10m": [5.0, 5.0, None], "wind_direction_10m": [270.0, 270.0, 270.0]}
            units = {"wind_speed_10m": "m/s"}
        else:
            assert url == ea.OPENMETEO_MARINE
            h = {"time": hours, "ocean_current_velocity": [3.6, 3.6, 3.6], "ocean_current_direction": [0.0, 0.0, 0.0]}
            units = {"ocean_current_velocity": "km/h"}
        return [{"hourly": h, "hourly_units": units} for _ in range(n)]
    return fetch


def test_period_and_file_name():
    assert ea.period(WHEN) == (date(2023, 5, 9), date(2023, 5, 13))
    name = ea.auto_name("wind", "openmeteo", BBOX, *ea.period(WHEN))
    assert name.startswith("wind_openmeteo_20230509-20230513_") and name.endswith(".nc")


def test_wind_from_west_blows_east_and_current_units():
    t, lats, lons, u, v = ea.openmeteo_grid("wind", BBOX, *ea.period(WHEN), fetch=fake_openmeteo("wind"))
    assert u.shape == (3, 3, 3) and len(lats) == 3 and len(lons) == 3
    assert u[0, 1, 1] == pytest.approx(5.0) and abs(v[0, 1, 1]) < 1e-9   # wind FROM 270 deg = towards east
    assert np.isnan(u[2, 0, 0])                                           # missing value stays missing
    _, _, _, cu, cv = ea.openmeteo_grid("current", BBOX, *ea.period(WHEN), fetch=fake_openmeteo("current"))
    assert cv[0, 0, 0] == pytest.approx(1.0) and abs(cu[0, 0, 0]) < 1e-9  # 3.6 km/h TOWARDS north = 1 m/s north


def test_file_must_cover_place_and_time():
    huntington = (33.0, 34.1, -118.8, -117.4, datetime(2021, 10, 1), datetime(2021, 10, 6))
    assert not ea.covers(huntington, BBOX, WHEN)                          # California file is not used for Mumbai
    hb_box = (-118.25, 33.42, -117.85, 33.72)
    assert ea.covers(huntington, hb_box, datetime(2021, 10, 3, 1, tzinfo=timezone.utc))
    assert not ea.covers(huntington, hb_box, datetime(2022, 3, 1, tzinfo=timezone.utc))   # wrong year
    assert not ea.covers(None, hb_box, WHEN)


def test_prepare_uses_typed_values_and_reports_why(monkeypatch):
    monkeypatch.setattr(ea, "find_covering", lambda kind, bbox, when: None)
    monkeypatch.setattr(ea, "_wind", lambda b, w, log: (None, "wind could not be downloaded: default wind used"))
    monkeypatch.setattr(ea, "has_cmems", lambda: False)
    old = datetime(2021, 10, 3, 1, tzinfo=timezone.utc)
    res = ea.prepare((-118.25, 33.42, -117.85, 33.72), old, need_wind=True, need_current=True, log=lambda m: None)
    assert res["wind_file"] is None and res["current_file"] is None
    assert any("default wind" in n for n in res["notes"])
    assert any("before 2022" in n for n in res["notes"])
    typed = ea.prepare(BBOX, WHEN, need_wind=False, need_current=False)
    assert typed["notes"] == ["wind: value typed by the analyst", "current: value typed by the analyst"]
