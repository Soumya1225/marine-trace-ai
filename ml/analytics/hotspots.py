"""Pollution hotspots and trends from all incidents (GIS analysis, as in Ivanov & Zatyagalova).

Hotspots: incidents are grouped into grid cells of CELL_DEG (about 10 km). Each cell gets the number of
spills, total area, highest priority and the nearest named coastal site, and a hotspot score that ranks
cells by how often and how much oil was found there. Incidents marked 'rejected' (false alarms) are left out.

Trend: per satellite-image date, number of spills and area, split by the look-alike check verdict.
"""
import math
from collections import defaultdict

CELL_DEG = 0.1
PRIORITY_RANK = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}


# Coastal towns and ports used to name hotspots that are far from any listed sensitive site
# (approximate positions; the sensitive-sites file only covers the case-study coast).
COASTAL_PLACES = [
    ("Kochi", 9.97, 76.27), ("Alappuzha", 9.49, 76.33), ("Kollam", 8.89, 76.61), ("Thiruvananthapuram", 8.52, 76.94),
    ("Kozhikode", 11.25, 75.78), ("Kannur", 11.87, 75.37), ("Mangaluru", 12.91, 74.86), ("Karwar", 14.81, 74.13),
    ("Panaji (Goa)", 15.49, 73.83), ("Ratnagiri", 16.99, 73.30), ("Mumbai", 18.94, 72.84), ("Dahej", 21.70, 72.58),
    ("Porbandar", 21.64, 69.61), ("Okha", 22.47, 69.07), ("Kandla", 23.03, 70.22), ("Kanyakumari", 8.08, 77.55),
    ("Thoothukudi", 8.76, 78.13), ("Nagapattinam", 10.77, 79.84), ("Puducherry", 11.93, 79.83), ("Chennai", 13.08, 80.29),
    ("Machilipatnam", 16.17, 81.13), ("Kakinada", 16.95, 82.25), ("Visakhapatnam", 17.69, 83.30), ("Gopalpur", 19.26, 84.91),
    ("Puri", 19.80, 85.83), ("Paradip", 20.26, 86.67), ("Digha", 21.63, 87.55), ("Haldia", 22.03, 88.06),
    ("Port Blair", 11.67, 92.73), ("Long Beach", 33.77, -118.19), ("Huntington Beach", 33.66, -118.00),
    ("Newport Beach", 33.62, -117.93), ("Laguna Beach", 33.54, -117.78), ("Dana Point", 33.46, -117.70),
]
OPEN_SEA_KM = 150


def nearest_place(lat, lon, sites=()):
    """Name of the closest sensitive site or coastal town, e.g. 'Alappuzha (45 km)'."""
    cands = [(s["name"], s["lat"], s["lon"]) for s in sites] + COASTAL_PLACES
    name, plat, plon = min(cands, key=lambda c: _km(lat, lon, c[1], c[2]))
    d = _km(lat, lon, plat, plon)
    return f"{name} ({d:.0f} km)" if d <= OPEN_SEA_KM else f"Open sea, {d:.0f} km from {name}"


def _km(lat1, lon1, lat2, lon2):
    dy = (lat2 - lat1) * 111.32
    dx = (lon2 - lon1) * 111.32 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dx, dy)


def hotspots(incidents, sites=(), top=10):
    """incidents: dicts with lat, lon, area_km2, priority, status, verdict (optional)."""
    cells = defaultdict(list)
    for i in incidents:
        if i.get("status") == "rejected" or i.get("lat") is None:
            continue
        cells[(math.floor(i["lat"] / CELL_DEG), math.floor(i["lon"] / CELL_DEG))].append(i)
    out = []
    for (r, c), items in cells.items():
        lat = sum(x["lat"] for x in items) / len(items)
        lon = sum(x["lon"] for x in items) / len(items)
        area = sum(x.get("area_km2") or 0 for x in items)
        oil = sum(1 for x in items if x.get("verdict") != "Likely look-alike")
        worst = max(items, key=lambda x: PRIORITY_RANK.get(x.get("priority"), 0))["priority"]
        out.append({
            "lat": round(lat, 4), "lon": round(lon, 4),
            "bounds": [c * CELL_DEG, r * CELL_DEG, (c + 1) * CELL_DEG, (r + 1) * CELL_DEG],
            "count": len(items), "likely_oil": oil, "area_km2": round(area, 3), "max_priority": worst,
            "incident_ids": sorted(x["id"] for x in items),
            "near": nearest_place(lat, lon, sites),
            "score": round(oil * 1.0 + (len(items) - oil) * 0.3 + math.log1p(area) + PRIORITY_RANK.get(worst, 0) * 0.5, 2),
        })
    out.sort(key=lambda h: -h["score"])
    return out[:top]


def trend(incidents):
    """Per image date: count and area of likely-oil vs likely look-alike spills."""
    days = defaultdict(lambda: {"oil": 0, "lookalike": 0, "area_km2": 0.0})
    for i in incidents:
        if i.get("status") == "rejected" or not i.get("date"):
            continue
        d = days[i["date"]]
        d["lookalike" if i.get("verdict") == "Likely look-alike" else "oil"] += 1
        d["area_km2"] += i.get("area_km2") or 0
    return [{"date": k, **v, "area_km2": round(v["area_km2"], 3), "count": v["oil"] + v["lookalike"]}
            for k, v in sorted(days.items())]
