"""Offshore infrastructure layer: oil platforms and pipelines (GIS context for source attribution).

Not every spill comes from a ship: platforms leak and pipelines break (the October 2021 Huntington Beach
spill came from pipeline P00547, not from a passing tanker). This module

  * loads platform and pipeline positions from ml/gis/offshore_infrastructure.geojson,
  * measures how far each spill (outline and back-tracked drift path) is from them,
  * flags "possible non-ship source" when a spill or its drift path comes close to a platform or pipeline,
  * marks radar targets that sit on a known platform as fixed structures instead of dark vessels.

Ship suspects are still listed; the flag tells the investigator to check the installation as well.
"""
import json
import math
from pathlib import Path

LAYER = Path(__file__).resolve().parents[1] / "gis" / "offshore_infrastructure.geojson"
SEARCH_KM = 15.0          # list installations within this distance of the spill
ON_SOURCE_KM = 1.0        # spill outline touches / covers the installation
NEAR_SOURCE_KM = 3.0      # spill or its back-tracked drift passes this close
PLATFORM_TARGET_KM = 0.5  # a radar target this close to a known platform is the platform itself

_KM_LAT = 111.32


def load_layer(path=LAYER):
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("features", [])


def _xy(lon, lat, lat0):
    return lon * _KM_LAT * math.cos(math.radians(lat0)), lat * _KM_LAT


def _seg_dist(p, a, b):
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _inside(p, ring):
    x, y, inside = p[0], p[1], False
    for i in range(len(ring)):
        (x1, y1), (x2, y2) = ring[i], ring[i - 1]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1:
            inside = not inside
    return inside


def _lines(coords):
    """Point -> one zero-length segment; LineString -> its segments; Polygon ring -> its edges."""
    if len(coords) == 1:
        return [(coords[0], coords[0])]
    return list(zip(coords[:-1], coords[1:]))


def _distance_km(geom_a, geom_b, lat0, ring_a=None):
    """Shortest distance (km) between two lists of lon/lat points treated as polylines.
    If ring_a is given (spill outline), anything inside it is at distance 0."""
    A = [_xy(lo, la, lat0) for lo, la in geom_a]
    B = [_xy(lo, la, lat0) for lo, la in geom_b]
    if ring_a and len(ring_a) >= 3:
        R = [_xy(lo, la, lat0) for lo, la in ring_a]
        if any(_inside(p, R) for p in B):
            return 0.0
    best = float("inf")
    for p in A:
        for a, b in _lines(B):
            best = min(best, _seg_dist(p, a, b))
    for p in B:
        for a, b in _lines(A):
            best = min(best, _seg_dist(p, a, b))
    return best


def _coords(feature):
    g = feature["geometry"]
    return [g["coordinates"]] if g["type"] == "Point" else list(g["coordinates"])


def assess_spill(spill, backtrack=None, layer=None):
    """Distances from the spill (outline + back-tracked drift) to each installation and the source verdict."""
    layer = load_layer() if layer is None else layer
    lat0 = spill["centroid_lat"]
    outline = [tuple(p) for p in (spill.get("polygon") or [])] or [(spill["centroid_lon"], spill["centroid_lat"])]
    drift = [tuple(p[:2]) for p in (backtrack or [])]
    nearby = []
    for f in layer:
        c = _coords(f)
        d_spill = _distance_km(outline, c, lat0, ring_a=outline)
        d_drift = _distance_km(drift, c, lat0) if len(drift) >= 2 else None
        d = min(d_spill, d_drift) if d_drift is not None else d_spill
        if d <= SEARCH_KM:
            p = f["properties"]
            nearby.append({"name": p["name"], "kind": p["kind"], "operator": p.get("operator", ""),
                           "role": p.get("role", ""), "distance_km": round(d_spill, 2),
                           "drift_distance_km": round(d_drift, 2) if d_drift is not None else None,
                           "geometry": f["geometry"]})
    nearby.sort(key=lambda x: min(x["distance_km"], x["drift_distance_km"] if x["drift_distance_km"] is not None else 1e9))

    level, note = "none", "No oil platform or pipeline within %.0f km of the spill or its drift path." % SEARCH_KM
    if nearby:
        n = nearby[0]
        closest = min(n["distance_km"], n["drift_distance_km"] if n["drift_distance_km"] is not None else 1e9)
        where = "the spill outline" if n["distance_km"] <= closest + 1e-9 else "the back-tracked drift path"
        if n["distance_km"] <= ON_SOURCE_KM:
            level = "likely"
            note = (f"The spill lies on or next to {n['name']} ({n['distance_km']} km). A leak from this "
                    f"{n['kind']} is a likely source: inspect it before blaming a ship.")
        elif closest <= NEAR_SOURCE_KM:
            level = "possible"
            note = (f"{where[0].upper() + where[1:]} passes {closest} km from {n['name']}. The oil may come from this "
                    f"{n['kind']} rather than from a ship: check it together with the ship suspects.")
        else:
            note = f"Nearest installation: {n['name']}, {closest} km away. A ship is the more likely source."
    return {"possible_non_ship_source": level != "none", "level": level, "note": note, "nearby": nearby[:6],
            "layer": "ml/gis/offshore_infrastructure.geojson (approximate positions)"}


def action_for(info):
    """Extra recommended action when an installation may be the source."""
    if not info or not info.get("possible_non_ship_source"):
        return None
    n = info["nearby"][0]
    what = ("check pipeline pressure / flow records and send an ROV or diver survey along the line"
            if n["kind"] == "pipeline" else "contact the platform operator and check for leaks, spills or discharges")
    return {"urgency": "high" if info["level"] == "likely" else "medium",
            "action": f"Inspect {n['name']} (possible non-ship source)",
            "detail": f"{info['note']} Operator: {n['operator'] or 'unknown'}; {what}."}


def mark_known_platforms(targets, layer=None, radius_km=PLATFORM_TARGET_KM):
    """Radar targets sitting on a known platform are the platform, not a dark ship. Returns how many were marked."""
    layer = load_layer() if layer is None else layer
    plats = [f for f in layer if f["geometry"]["type"] == "Point"]
    n = 0
    for t in targets:
        if not (t.get("dark") or t.get("fixed")):   # matched to an AIS ship (e.g. a supply boat at the platform)
            continue
        for f in plats:
            lon, lat = f["geometry"]["coordinates"]
            dy = (t["lat"] - lat) * _KM_LAT
            dx = (t["lon"] - lon) * _KM_LAT * math.cos(math.radians(lat))
            if math.hypot(dx, dy) <= radius_km:
                if not t.get("fixed"):
                    n += 1
                t["dark"] = False
                t["fixed"] = True
                t["fixed_note"] = f"matches the known position of {f['properties']['name']}"
                break
    return n
