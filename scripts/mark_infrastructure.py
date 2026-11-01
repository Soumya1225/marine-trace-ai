"""Add the oil platform / pipeline check to analyses that were run before it existed.

For every saved result and every incident in the database:
  * radar targets on a known platform become fixed structures (not dark vessels),
  * each spill gets an 'infrastructure' block (nearby platforms/pipelines, possible non-ship source),
  * an 'Inspect <platform/pipeline>' action is added when an installation may be the source.

    python scripts/mark_infrastructure.py        (run from the project folder; backend may keep running)
"""
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from ml.features.fixed_structures import ship_counts  # noqa: E402
from ml.features.infrastructure import action_for, assess_spill, load_layer, mark_known_platforms  # noqa: E402
from ml.features.ship_detection import dark_vessels_near  # noqa: E402

RESULTS = ROOT / "backend" / "uploads" / "results"
TAG = "(possible non-ship source)"


def update_actions(decision, info):
    acts = [a for a in (decision.get("actions") or []) if TAG not in a.get("action", "")]
    extra = action_for(info)
    if extra:
        acts.insert(1, extra)
    decision["actions"] = acts


def main():
    layer = load_layer()
    print(f"Infrastructure layer: {sum(f['geometry']['type'] == 'Point' for f in layer)} platforms, "
          f"{sum(f['geometry']['type'] == 'LineString' for f in layer)} pipelines")
    per_time = {}
    for p in sorted(RESULTS.glob("*/result.json")):
        r = json.loads(p.read_text())
        targets = (r.get("ships") or {}).get("targets") or []
        n = mark_known_platforms(targets, layer)
        r["ships"] = {**ship_counts(targets), "targets": targets}
        for item in r.get("spills", []):
            sp = item["spill"]
            sp["nearby_dark_vessels"] = dark_vessels_near(targets, sp["centroid_lat"], sp["centroid_lon"])
            sp["infrastructure"] = assess_spill(sp, item.get("backtrack"), layer)
            update_actions(item["decision"], sp["infrastructure"])
        p.write_text(json.dumps(r, indent=2, default=str))
        per_time[r["scene"]["acquisition_time"]] = r["ships"]
        print(f"{r['scene']['source'][-40:]}: {n} radar targets matched to known platforms")

    os.chdir(ROOT / "backend")              # the database path is relative to the backend folder
    from app.database import SessionLocal
    from app.models import Incident
    db = SessionLocal()
    try:
        for inc in db.query(Incident).order_by(Incident.id).all():
            d = copy.deepcopy(inc.data or {})
            sp = d.get("spill") or {}
            if "centroid_lat" not in sp:
                continue
            when = str(sp.get("acquisition_time"))
            ships = per_time.get(when) or next((v for k, v in per_time.items() if k[:16] == when[:16]), None)
            if ships is not None:
                d["ships"] = copy.deepcopy(ships)
            targets = (d.get("ships") or {}).get("targets") or []
            mark_known_platforms(targets, layer)
            if d.get("ships"):
                d["ships"] = {**ship_counts(targets), "targets": targets}
            sp["nearby_dark_vessels"] = dark_vessels_near(targets, sp["centroid_lat"], sp["centroid_lon"])
            sp["infrastructure"] = assess_spill(sp, d.get("backtrack"), layer)
            d["spill"] = sp
            if d.get("decision"):
                update_actions(d["decision"], sp["infrastructure"])
            inc.data = d
            info = sp["infrastructure"]
            near = info["nearby"][0] if info["nearby"] else None
            print(f"Incident #{inc.id}: " + (f"nearest {near['name']} {near['distance_km']} km" if near else "nothing within 15 km")
                  + (f"  ->  {info['level'].upper()} non-ship source" if info["possible_non_ship_source"] else ""))
        db.commit()
    finally:
        db.close()
    print("Done. Refresh an incident page: platforms and pipelines are on the map (layer 'Oil platforms & pipelines').")


if __name__ == "__main__":
    main()
