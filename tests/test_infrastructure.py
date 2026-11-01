"""Platform / pipeline layer: possible non-ship source flag and known-platform marking."""
from ml.features.infrastructure import action_for, assess_spill, load_layer, mark_known_platforms


def _square(lon, lat, d=0.004):
    return [[lon - d, lat - d], [lon + d, lat - d], [lon + d, lat + d], [lon - d, lat + d]]


def test_layer_loads():
    layer = load_layer()
    assert any(f["properties"]["kind"] == "platform" for f in layer)
    assert any(f["properties"]["kind"] == "pipeline" for f in layer)


def test_spill_on_pipeline_is_flagged():
    spill = {"centroid_lat": 33.64, "centroid_lon": -118.095, "polygon": _square(-118.095, 33.64)}
    info = assess_spill(spill)
    assert info["possible_non_ship_source"] and info["level"] == "likely"
    assert "P00547" in info["nearby"][0]["name"]
    assert action_for(info)["action"].startswith("Inspect")


def test_far_spill_is_not_flagged():
    spill = {"centroid_lat": 33.40, "centroid_lon": -118.50, "polygon": _square(-118.50, 33.40)}
    info = assess_spill(spill)
    assert not info["possible_non_ship_source"] and action_for(info) is None


def test_drift_path_near_platform_is_possible():
    spill = {"centroid_lat": 33.53, "centroid_lon": -118.20, "polygon": _square(-118.20, 33.53)}
    back = [[-118.20, 33.53], [-118.16, 33.56], [-118.135, 33.578]]      # drifted from near Platform Elly
    info = assess_spill(spill, back)
    assert info["level"] == "possible"


def test_dark_target_on_platform_becomes_fixed():
    targets = [{"lat": 33.5823, "lon": -118.1276, "dark": True}, {"lat": 33.45, "lon": -118.4, "dark": True},
               {"lat": 33.5640, "lon": -118.1166, "dark": False, "vessel_name": "SUPPLY BOAT"}]
    assert mark_known_platforms(targets) == 1
    assert targets[0]["fixed"] and not targets[0]["dark"]
    assert targets[1]["dark"] and not targets[2].get("fixed")
