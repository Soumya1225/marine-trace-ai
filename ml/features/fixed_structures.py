"""Tell fixed structures (oil platforms) apart from ships without AIS.

A ship moves; an oil platform does not. When the same area has been analysed at another time (hours apart),
a bright radar target that sits at the same spot in both images is a fixed structure (or a ship at anchor),
not a "dark vessel". Such targets get fixed=True and dark=False, so they leave the dark-vessel lists.
"""
import json
from datetime import datetime
from pathlib import Path

from ml.utils.geo import haversine_km

MIN_GAP_H = 3.0      # images must be at least this far apart in time
RADIUS_KM = 0.25     # same spot = within 250 m (radar position error plus platform size)


def _overlaps(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def other_scene_targets(results_dir, image_time, bounds, exclude_dir=None):
    """Radar targets of other analyses of the same area taken at least MIN_GAP_H hours away."""
    out = []
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return out
    for p in results_dir.glob("*/result.json"):
        if exclude_dir is not None and p.parent.resolve() == Path(exclude_dir).resolve():
            continue
        try:
            r = json.loads(p.read_text())
            t = datetime.fromisoformat(r["scene"]["acquisition_time"])
            if abs((t - image_time).total_seconds()) / 3600 < MIN_GAP_H:
                continue
            if not _overlaps(r["scene"]["bounds"], bounds):
                continue
            targets = (r.get("ships") or {}).get("targets") or []
            if targets:
                out.append({"time": t.isoformat(), "source": r["scene"]["source"], "targets": targets})
        except Exception:
            continue
    return out


def mark_fixed(targets, others, radius_km=RADIUS_KM):
    """Mark targets that also appear at the same spot in another image. Returns how many were marked."""
    n = 0
    for t in targets:
        for other in others:
            if any(haversine_km(t["lat"], t["lon"], o["lat"], o["lon"]) <= radius_km for o in other["targets"]):
                if t.get("dark"):
                    t["dark"] = False
                    t["fixed"] = True
                    t["fixed_note"] = f"also at this spot in the image of {other['time'][:16].replace('T', ' ')} UTC"
                    n += 1
                break
    return n


def ship_counts(targets):
    fixed = sum(1 for x in targets if x.get("fixed"))
    dark = sum(1 for x in targets if x.get("dark"))
    return {"detected": len(targets), "dark": dark, "fixed": fixed, "matched": len(targets) - dark - fixed}
