"""Tests for geometry, spill features, drift and impact (no GPU / internet needed).
Run from the project root:  python -m pytest tests -q
"""
from datetime import datetime, timezone

import numpy as np

from ml.attribution.drift import Environment, seed_particles, simulate
from ml.features.spill_features import describe_points, extract_spills
from ml.impact.assessment import assess, volume_estimate
from ml.utils.geo import axis_angle_diff, haversine_km, point_to_polyline_km, transform_from_bbox


def test_haversine_one_degree_latitude():
    assert abs(haversine_km(0, 0, 1, 0) - 111.19) < 0.2


def test_axis_angle_diff_is_undirected():
    assert axis_angle_diff(10, 190) == 0
    assert axis_angle_diff(0, 90) == 90


def test_point_to_polyline():
    d = point_to_polyline_km([0.0], [1.0], [-1.0, 1.0], [0.0, 0.0])
    assert abs(d[0] - 1.0) < 1e-9


def test_describe_points_orientation_east_west():
    lon = np.linspace(-118.1, -118.0, 200)
    lat = np.full_like(lon, 33.6) + np.random.default_rng(0).normal(0, 1e-4, 200)
    d = describe_points(lat, lon)
    assert abs(d["orientation_deg"] - 90) < 5        # long axis points east-west
    assert d["elongation"] > 5


def test_extract_spills_finds_one_blob():
    cm = np.zeros((200, 300), np.uint8)
    cm[90:110, 50:250] = 1
    t = transform_from_bbox((-118.2, 33.5, -117.9, 33.7), 300, 200)
    spills = extract_spills(cm, t, datetime(2021, 10, 3, tzinfo=timezone.utc))
    assert len(spills) == 1
    assert spills[0]["area_km2"] > 0
    assert len(spills[0]["polygon"]) >= 4


def test_drift_moves_with_wind():
    env = Environment(wind_u=10, wind_v=0, current_u=0, current_v=0)
    lat, lon = seed_particles([[-118.0, 33.6]], n=100, jitter_km=0.01)
    snaps = simulate(lat, lon, datetime(2021, 10, 3, tzinfo=timezone.utc), 10, env, diffusion=0.0)
    moved_km = (np.mean(snaps[-1]["lon"]) - np.mean(snaps[0]["lon"])) * 111.32 * np.cos(np.radians(33.6))
    assert abs(moved_km - 0.3 * 10 * 3.6) < 0.5     # 3% of 10 m/s for 10 h = 10.8 km east


def test_backward_is_reverse_of_forward():
    env = Environment(wind_u=0, wind_v=5, current_u=0.1, current_v=0)
    t0 = datetime(2021, 10, 3, tzinfo=timezone.utc)
    fwd = simulate([33.6], [-118.0], t0, 6, env, diffusion=0.0)
    back = simulate(fwd[-1]["lat"], fwd[-1]["lon"], t0, 6, env, direction=-1, diffusion=0.0)
    assert haversine_km(33.6, -118.0, back[-1]["lat"][0], back[-1]["lon"][0]) < 0.05


def test_volume_estimate_units():
    v = volume_estimate(1.0)            # 1 km2 at 0.3..50 um
    assert v["m3"] == [0.3, 50.0]


def test_assess_detects_threatened_site():
    spill = {"area_km2": 2.0}
    snaps = [{"hours": h, "lat": np.full(50, 33.60 + 0.01 * h), "lon": np.full(50, -118.0),
              "beached": np.zeros(50, bool)} for h in range(6)]
    sites = [{"name": "Test beach", "type": "beach", "lat": 33.65, "lon": -118.0, "sensitivity": 4}]
    out = assess(spill, snaps, sites)
    assert out["threatened_sites"][0]["name"] == "Test beach"
