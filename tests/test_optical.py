"""Tests for the Sentinel-2 optical check (A6), with a fake catalogue (no network).
    python -m pytest tests/test_optical.py -q
"""
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as N

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services import optical  # noqa: E402

RADAR = datetime(2025, 5, 28, 0, 41, tzinfo=timezone.utc)


def product(name, start, cloud, ql="a1b2-c3"):
    return {"Name": name, "ContentDate": {"Start": start},
            "Attributes": [{"Name": "cloudCover", "Value": cloud}],
            "Assets": [{"Type": "QUICKLOOK", "Id": ql}] if ql else []}


class Resp:
    def __init__(self, data):
        self.data, self.status_code, self.content = data, 200, b"jpg"

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def fake_get(url, params=None, timeout=None, headers=None):
    assert "S2MSI2A" in url and "cloudCover" in url and " and " in url    # spaces kept (sent as %20, not +)
    return Resp({"value": [
        product("S2A_MSIL2A_20250531T052651_N0511_R105_T43PEN_20250531T090111.SAFE", "2025-05-31T05:26:51.000Z", 85.0),
        product("S2B_MSIL2A_20250528T052649_N0511_R105_T43PEN_20250528T080000.SAFE", "2025-05-28T05:26:49.000Z", 12.5),
        product("S2B_MSIL2A_20250528T052649_N0500_R105_T43PEN_20250528T070000.SAFE", "2025-05-28T05:26:49.000Z", 12.5),
    ]})


def test_search_dedupes_and_sorts_by_time_gap():
    scenes = optical.search(9.498, 75.762, RADAR, get=fake_get)
    assert len(scenes) == 2                                        # reprocessed duplicate removed
    assert scenes[0]["hours_from_radar"] == pytest.approx(4.8, abs=0.1) and scenes[0]["cloud_pct"] == 12.5
    assert scenes[0]["tile"] == "T43PEN" and "S2_L2A_CDAS" in scenes[0]["browser_url"]


def test_incident_note_prefers_clear_images():
    inc = N(centroid_lat=9.498, centroid_lon=75.762, detected_at=RADAR.replace(tzinfo=None), created_at=None)
    res = optical.for_incident(inc, get=fake_get)
    assert res["note"].startswith("Best image") and "12.5% cloud" in res["note"]


def test_quicklook_id_is_checked():
    with pytest.raises(ValueError):
        optical.quicklook("../../etc/passwd")
