"""Impact assessment: spill volume range, shoreline at risk and threatened sensitive sites."""
import csv
import json
from pathlib import Path

import numpy as np

from ml.config import (COASTLINE_PATH, SENSITIVE_SITES_PATH, SITE_THREAT_KM,
                       THICKNESS_UM_HIGH, THICKNESS_UM_LOW)
from ml.utils.geo import haversine_km

M3_TO_BARRELS = 6.2898
M3_TO_GALLONS = 264.172


def load_coastline(path=COASTLINE_PATH, bbox=None, spacing_km=0.25):
    """Load a coastline GeoJSON (LineStrings) as densified point arrays (lat, lon)."""
    path = Path(path)
    if not path.exists():
        return np.array([]), np.array([])
    gj = json.loads(path.read_text())
    lats, lons = [], []
    feats = gj["features"] if "features" in gj else [gj]
    for feat in feats:
        geom = feat.get("geometry", feat) or {}
        if geom.get("type") == "LineString":
            lines = [geom["coordinates"]]
        elif geom.get("type") in ("MultiLineString", "Polygon"):
            lines = geom["coordinates"]
        elif geom.get("type") == "MultiPolygon":
            lines = [ring for poly in geom["coordinates"] for ring in poly]
        else:
            continue
        for line in lines:
            pts = np.asarray(line, float)
            for (lo1, la1), (lo2, la2) in zip(pts[:-1], pts[1:]):
                d = float(haversine_km(la1, lo1, la2, lo2))
                n = max(1, int(d / spacing_km))
                t = np.linspace(0, 1, n, endpoint=False)
                lats.extend(la1 + t * (la2 - la1))
                lons.extend(lo1 + t * (lo2 - lo1))
    lats, lons = np.array(lats), np.array(lons)
    if bbox is not None and len(lats):
        min_lon, min_lat, max_lon, max_lat = bbox
        keep = (lons >= min_lon) & (lons <= max_lon) & (lats >= min_lat) & (lats <= max_lat)
        lats, lons = lats[keep], lons[keep]
    return lats, lons


def load_sites(path=SENSITIVE_SITES_PATH):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["lat"], r["lon"] = float(r["lat"]), float(r["lon"])
        r["sensitivity"] = int(r.get("sensitivity") or 3)
    return rows


def volume_estimate(area_km2):
    """volume (m^3) = area (km^2) * thickness (um), because 1e6 m^2 * 1e-6 m = 1 m^3."""
    low, high = area_km2 * THICKNESS_UM_LOW, area_km2 * THICKNESS_UM_HIGH
    return {
        "thickness_um_range": [THICKNESS_UM_LOW, THICKNESS_UM_HIGH],
        "m3": [round(low, 2), round(high, 2)],
        "barrels": [round(low * M3_TO_BARRELS, 1), round(high * M3_TO_BARRELS, 1)],
        "gallons": [round(low * M3_TO_GALLONS), round(high * M3_TO_GALLONS)],
    }


def assess(spill, snapshots, sites, coast_spacing_km=0.25):
    """snapshots: forward drift output (with 'beached' flags)."""
    first_landfall_h, beached_frac = None, 0.0
    for s in snapshots:
        frac = float(np.mean(s["beached"]))
        if frac > 0.01 and first_landfall_h is None:
            first_landfall_h = round(s["hours"], 1)
        beached_frac = frac
    final = snapshots[-1]
    b = final["beached"]
    coast_km = 0.0
    if b.any():
        cells = {(round(la / 0.0025), round(lo / 0.0025)) for la, lo in zip(final["lat"][b], final["lon"][b])}
        coast_km = round(len(cells) * coast_spacing_km, 2)

    threatened = []
    for site in sites:
        eta, min_d = None, float("inf")
        for s in snapshots:
            d = haversine_km(site["lat"], site["lon"], s["lat"], s["lon"])
            dmin = float(np.min(d))
            min_d = min(min_d, dmin)
            if eta is None and np.mean(d <= SITE_THREAT_KM) > 0.02:
                eta = round(s["hours"], 1)
        if eta is not None:
            threatened.append({**site, "eta_h": eta, "min_distance_km": round(min_d, 2)})
    threatened.sort(key=lambda x: (x["eta_h"], -x["sensitivity"]))

    vol = volume_estimate(spill["area_km2"])
    severity = min(100.0, (
        25 * min(spill["area_km2"] / 20.0, 1.0)
        + 25 * (1.0 if first_landfall_h is not None and first_landfall_h <= 24 else 0.5 if first_landfall_h else 0.0)
        + 30 * min(sum(t["sensitivity"] for t in threatened) / 15.0, 1.0)
        + 20 * min(beached_frac * 2, 1.0)
    ))
    return {
        "volume": vol,
        "first_landfall_h": first_landfall_h,
        "beached_fraction_72h": round(beached_frac, 3),
        "coastline_affected_km": coast_km,
        "threatened_sites": threatened,
        "severity_score": round(severity, 1),
    }
