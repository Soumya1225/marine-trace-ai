"""Tests for the drifter validation (C6). Fake drifter data and a fake constant current: no network needed.
    python -m pytest tests/test_drifters.py -q
"""
from datetime import datetime, timedelta, timezone

import numpy as np

from ml.validation import drifters as dv

T0 = datetime(2022, 4, 1, tzinfo=timezone.utc)


def fake_csv(hours=100, gap_at=None):
    lines = ["ID,time,latitude,longitude", ",UTC,degrees_north,degrees_east"]
    for h in range(hours):
        if gap_at is not None and gap_at <= h < gap_at + 6:
            continue
        t = T0 + timedelta(hours=h)
        lon = 70.0 + 0.5 * 3600 * h / 1000 / 109.5          # 0.5 m/s towards east (km per degree lon at 10N)
        lines.append(f"1001,{t:%Y-%m-%dT%H:%M:%SZ},10.0,{lon:.5f}")
    return "\n".join(lines)


class ConstantCurrent:
    def at(self, lat, lon, when):
        return 0.0, 0.0, 0.5, 0.0                            # wind_u, wind_v, current_u, current_v


def test_parse_and_segments():
    pts = dv.parse_csv(fake_csv())
    assert len(pts) == 100 and pts[0]["id"] == "1001"
    segs = dv.segments(pts)
    assert len(segs) == 2                                     # starts at 0 h and 24 h fit 72 h windows in 100 h
    assert dv.segments(dv.parse_csv(fake_csv(gap_at=30))) == []        # a 6 h gap breaks every 72 h window


def test_model_matches_drifter_in_known_current():
    segs = dv.segments(dv.parse_csv(fake_csv()))
    res = dv.evaluate(segs, env_fn=lambda s: ConstantCurrent(), verbose=False)
    m = res["mean_separation_km"]
    assert res["segments_used"] == 2
    assert m["current_only"]["72h"] < 2.0                    # follows the drifter
    assert m["no_movement"]["72h"] > 100                     # 0.5 m/s for 72 h = ~130 km
    assert res["skill"]["current_only"] > 0.95 and res["skill"]["no_movement"] < 0.1
    assert "better" in dv.verdict(res)


def test_skill_score_edges():
    assert dv.skill([0, 0, 0], [10, 20, 30]) == 1.0
    assert dv.skill([100, 100, 100], [10, 20, 30]) == 0.0
    assert dv.skill([1], [0]) is None
