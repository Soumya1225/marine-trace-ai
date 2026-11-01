"""Update analyses that were run before fixed-structure detection existed.

Compares the radar targets of every analysed image with the other images of the same area, marks
targets that sit at the same spot hours apart as fixed structures (likely oil platforms), and updates
both the saved result files and the incidents in the database.

    python scripts/mark_fixed_structures.py        (run from the project folder; backend may keep running)
"""
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from ml.features.fixed_structures import mark_fixed, other_scene_targets, ship_counts  # noqa: E402
from ml.features.ship_detection import dark_vessels_near  # noqa: E402

RESULTS = ROOT / "backend" / "uploads" / "results"


def main():
    files = sorted(RESULTS.glob("*/result.json"))
    if len(files) < 2:
        print("Need at least two analysed images of the same area.")
        return
    originals = {p: json.loads(p.read_text()) for p in files}
    per_time = {}
    for p, r in originals.items():
        t = datetime.fromisoformat(r["scene"]["acquisition_time"])
        others = other_scene_targets(RESULTS, t, r["scene"]["bounds"], exclude_dir=p.parent)
        new = copy.deepcopy(r)
        targets = (new.get("ships") or {}).get("targets") or []
        for x in targets:                       # start clean so running twice gives the same result
            if x.pop("fixed", None):
                x["dark"] = True
                x.pop("fixed_note", None)
        mark_fixed(targets, others)
        new["ships"] = {**ship_counts(targets), "targets": targets}
        for item in new.get("spills", []):
            sp = item["spill"]
            sp["nearby_dark_vessels"] = dark_vessels_near(targets, sp["centroid_lat"], sp["centroid_lon"])
        p.write_text(json.dumps(new, indent=2, default=str))
        per_time[r["scene"]["acquisition_time"]] = new["ships"]
        c = new["ships"]
        print(f"{r['scene']['source'][-40:]}: {c['detected']} radar targets | AIS ships {c['matched']} | "
              f"dark (no AIS) {c['dark']} | fixed structures {c['fixed']}")

    import os
    os.chdir(ROOT / "backend")              # the database path is relative to the backend folder
    from app.database import SessionLocal
    from app.models import Incident
    db = SessionLocal()
    updated = 0
    try:
        for inc in db.query(Incident).all():
            d = copy.deepcopy(inc.data or {})
            sp = d.get("spill") or {}
            ships = per_time.get(str(sp.get("acquisition_time")))
            if ships is None:
                when = sp.get("acquisition_time")
                ships = next((v for k, v in per_time.items() if when and k[:16] == str(when)[:16]), None)
            if ships is None:
                continue
            d["ships"] = copy.deepcopy(ships)
            sp["nearby_dark_vessels"] = dark_vessels_near(d["ships"]["targets"], sp["centroid_lat"], sp["centroid_lon"])
            d["spill"] = sp
            inc.data = d
            updated += 1
        db.commit()
    finally:
        db.close()
    print(f"Updated {updated} incidents. Refresh an incident page to see the white 'fixed structure' circles.")


if __name__ == "__main__":
    main()
