"""Attribution tests: the vessel that sailed along the slick must rank first, and the full demo must work."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ml.attribution.attribute import attribute_spill
from ml.attribution.drift import Environment
from ml.features.ais_features import vessel_type_info
from ml.features.spill_features import describe_points

T_IMG = datetime(2021, 10, 3, 13, 52, tzinfo=timezone.utc)


def _track(mmsi, lat0, lon0, lat1, lon1, hours_before_start, hours_before_end, vtype=70):
    times = pd.date_range(T_IMG - timedelta(hours=hours_before_start), T_IMG - timedelta(hours=hours_before_end), freq="5min")
    f = np.linspace(0, 1, len(times))
    return pd.DataFrame({"mmsi": mmsi, "time": times, "lat": lat0 + f * (lat1 - lat0), "lon": lon0 + f * (lon1 - lon0),
                         "sog": 12.0, "cog": 90.0, "heading": 90.0, "vessel_name": f"V{mmsi}", "imo": None,
                         "vessel_type": vtype, "length": 200.0})


def test_culprit_ranks_first():
    env = Environment(wind_u=0, wind_v=0, current_u=0, current_v=0)
    # slick lies east-west on latitude 33.60 (no drift so it stays on the culprit's path)
    lon = np.linspace(-118.10, -118.00, 100)
    spill = describe_points(np.full(100, 33.60), lon)
    spill["area_km2"] = 1.0
    ais = pd.concat([
        _track(111111111, 33.60, -118.20, 33.60, -117.90, 5, 1, vtype=80),    # sails along the slick
        _track(222222222, 33.50, -118.05, 33.70, -118.05, 5, 1),              # crosses it at right angles
        _track(333333333, 33.75, -118.20, 33.75, -117.90, 5, 1),              # far to the north
    ], ignore_index=True)
    suspects, _ = attribute_spill(spill, ais, T_IMG, env)
    assert suspects[0]["mmsi"] == 111111111
    assert suspects[0]["alignment"] > 0.9


def test_vessel_type_info():
    assert vessel_type_info(84) == ("Tanker", 1.0)
    assert vessel_type_info(None)[0] == "Unknown"


def test_demo_pipeline_end_to_end(tmp_path):
    from ml.pipeline import analyze
    from ml.utils.demo_data import main as make_demo
    meta = make_demo(tmp_path / "demo")
    res = analyze(tmp_path / "demo" / "demo_sar.png", [tmp_path / "demo" / "demo_ais.csv"], bbox=tuple(meta["bbox"]),
                  acquisition_time=meta["acquisition_time"], out_dir=tmp_path / "run")
    assert res["spills"], "demo slick should be detected"
    # the trained model may also report the round look-alike patch, so check the spill(s) the culprit is linked to
    tops = [sp["suspects"][0]["mmsi"] for sp in res["spills"] if sp["suspects"]]
    assert meta["true_mmsi"] in tops, "the guilty tanker must be the top suspect of at least one spill"
    assert res["spills"][0]["forecast"] and res["spills"][0]["decision"]["actions"]


def test_demo_dark_vessel_and_uncertainty(tmp_path):
    from ml.pipeline import analyze
    from ml.utils.demo_data import main as make_demo
    from ml.utils.geo import haversine_km
    meta = make_demo(tmp_path / "demo")
    res = analyze(tmp_path / "demo" / "demo_sar.png", [tmp_path / "demo" / "demo_ais.csv"], bbox=tuple(meta["bbox"]),
                  acquisition_time=meta["acquisition_time"], out_dir=tmp_path / "run")
    dark = [t for t in res["ships"]["targets"] if t["dark"]]
    assert len(dark) == 1, "exactly the planted ship has no AIS"
    assert haversine_km(dark[0]["lat"], dark[0]["lon"], meta["dark_vessel"]["lat"], meta["dark_vessel"]["lon"]) < 0.5
    assert res["ships"]["matched"] >= 2   # ships very close to the real coastline are masked with the land
    assert res["spills"][0]["forecast"][2]["uncertainty_polygon"]
