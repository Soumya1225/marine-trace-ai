"""Response planning: nearest response bases, travel time, equipment needed and a cost estimate.

All numbers are planning estimates for a first response, not operational orders.
Station positions are approximate (port locations); the equipment rules of thumb and the unit costs are
assumptions that an analyst can change on the page (or in .env) to match local rates.
"""
import math
import os

# Approximate positions of Indian Coast Guard regional / district headquarters and pollution-response bases
# (ports where they are based), plus the US Coast Guard sector for the Huntington Beach case study.
STATIONS = [
    ("ICG Pollution Response Team, Mumbai", "Indian Coast Guard", 18.93, 72.84, "Pollution response team (West)"),
    ("ICG Pollution Response Team, Chennai", "Indian Coast Guard", 13.10, 80.30, "Pollution response team (East)"),
    ("ICG Pollution Response Team, Port Blair", "Indian Coast Guard", 11.67, 92.73, "Pollution response team (Andaman & Nicobar)"),
    ("ICG Pollution Response Team, Vadinar", "Indian Coast Guard", 22.47, 69.70, "Pollution response team (Gulf of Kutch)"),
    ("ICG Station Paradip", "Indian Coast Guard", 20.26, 86.67, "District HQ, Odisha"),
    ("ICG Station Haldia", "Indian Coast Guard", 22.03, 88.06, "District HQ, West Bengal"),
    ("ICG Station Visakhapatnam", "Indian Coast Guard", 17.69, 83.29, "District HQ, Andhra Pradesh"),
    ("ICG Station Kochi", "Indian Coast Guard", 9.97, 76.26, "District HQ, Kerala"),
    ("ICG Station Goa", "Indian Coast Guard", 15.40, 73.80, "District HQ, Goa"),
    ("ICG Station Porbandar", "Indian Coast Guard", 21.64, 69.61, "District HQ, Gujarat"),
    ("ICG Station New Mangalore", "Indian Coast Guard", 12.92, 74.81, "District HQ, Karnataka"),
    ("ICG Station Tuticorin", "Indian Coast Guard", 8.76, 78.18, "Station, Tamil Nadu"),
    ("USCG Sector Los Angeles - Long Beach", "US Coast Guard", 33.72, -118.27, "Case-study area (California)"),
]

DEFAULTS = {
    "vessel_speed_kn": float(os.getenv("RESPONSE_VESSEL_SPEED_KN", "15")),
    "route_factor": 1.3,                  # sea route is longer than the straight line
    "boom_factor": 1.5,                   # boom length = 1.5 x slick perimeter (allows for J- and U-sweeps)
    "skimmer_m3_per_h": float(os.getenv("RESPONSE_SKIMMER_M3H", "20")),
    "skimmers": 2,
    "recovery_efficiency": 0.3,           # share of the oil that can realistically be recovered at sea
    # example unit costs: replace with local rates (they vary widely by country, oil type and shoreline)
    "cost_per_m3_at_sea": float(os.getenv("COST_PER_M3_SEA", "5000")),
    "cost_per_km_shoreline": float(os.getenv("COST_PER_KM_SHORE", "150000")),
    "currency": os.getenv("COST_CURRENCY", "USD"),
}


def _km(lat1, lon1, lat2, lon2):
    r = math.pi / 180
    h = (math.sin((lat2 - lat1) * r / 2) ** 2 +
         math.cos(lat1 * r) * math.cos(lat2 * r) * math.sin((lon2 - lon1) * r / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(h))


def _perimeter_km(ring, area_km2):
    if ring and len(ring) > 2:
        return sum(_km(a[1], a[0], b[1], b[0]) for a, b in zip(ring, ring[1:] + ring[:1]))
    return 2 * math.sqrt(math.pi * max(area_km2, 0.0))          # circle with the same area


def plan(inc, overrides=None):
    p = dict(DEFAULTS)
    p.update({k: v for k, v in (overrides or {}).items() if v is not None})
    d = inc.data or {}
    sp, im = d.get("spill") or {}, d.get("impact") or {}
    lat, lon = inc.centroid_lat, inc.centroid_lon

    stations = []
    for name, org, slat, slon, role in STATIONS:
        straight = _km(lat, lon, slat, slon)
        sea_km = straight * p["route_factor"]
        hours = sea_km / (p["vessel_speed_kn"] * 1.852)
        stations.append({"name": name, "org": org, "role": role, "lat": slat, "lon": slon,
                         "distance_km": round(sea_km, 1), "eta_h": round(hours, 1)})
    stations.sort(key=lambda s: s["distance_km"])
    near = [s for s in stations if s["distance_km"] <= 1000]
    nearest = near[:3] or stations[:1]                       # far-away bases are not useful to list

    perim = _perimeter_km(sp.get("polygon") or [], inc.area_km2)
    boom_m = perim * 1000 * p["boom_factor"]
    vol = (im.get("volume") or {}).get("m3") or [0, 0]
    recoverable = [v * p["recovery_efficiency"] for v in vol]
    rate = p["skimmer_m3_per_h"] * p["skimmers"]
    skim_h = [round(v / rate, 1) if rate else None for v in recoverable]
    shore_km = im.get("coastline_affected_km") or 0
    landfall = im.get("first_landfall_h")
    first_eta = nearest[0]["eta_h"] if nearest else None
    in_time = landfall is None or (first_eta is not None and first_eta < landfall)

    sea_cost = [round(v * p["cost_per_m3_at_sea"]) for v in vol]
    shore_cost = round(shore_km * p["cost_per_km_shoreline"])
    total = [sea_cost[0] + shore_cost, sea_cost[1] + shore_cost]
    # if the response arrives before landfall, most shoreline cleaning can be avoided
    avoided = shore_cost if (landfall is not None and in_time) else 0

    return {
        "nearest": nearest,
        "all_stations": stations,
        "equipment": {
            "slick_perimeter_km": round(perim, 2),
            "boom_m": int(round(boom_m, -1)),
            "skimmers": p["skimmers"], "skimmer_m3_per_h": p["skimmer_m3_per_h"],
            "recoverable_m3": [round(v, 1) for v in recoverable], "skimming_hours": skim_h,
        },
        "timing": {"first_landfall_h": landfall, "first_unit_eta_h": first_eta, "arrives_before_landfall": in_time},
        "cost": {"currency": p["currency"], "at_sea": sea_cost, "shoreline": shore_cost, "total": total,
                 "avoidable_if_contained_at_sea": avoided, "shoreline_km": shore_km,
                 "volume_m3": [round(v, 1) for v in vol]},
        "assumptions": p,
        "note": ("Planning estimate only. Station positions are approximate; boom length uses 1.5 x the slick "
                 "perimeter; recovery at sea is assumed to be 30% efficient; unit costs are example values "
                 "and must be replaced with local rates."),
    }
