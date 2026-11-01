"""Add the second-stage look-alike check to incidents analysed before it existed.

Uses the saved radar image (sar.png) and detection mask (mask.png) of each analysis, so nothing is re-run.

    python scripts/add_lookalike_check.py        (run from the project folder; backend may keep running)
"""
import copy
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from ml.lookalike.second_stage import check_spills  # noqa: E402


def load(result_dir, bounds):
    img = np.asarray(Image.open(Path(result_dir) / "sar.png").convert("L"))
    mask = np.asarray(Image.open(Path(result_dir) / "mask.png").convert("RGBA"))
    oil = (mask[..., 0] > 200) & (mask[..., 1] < 100) & (mask[..., 3] > 150)       # red = oil
    h, w = img.shape
    min_lon, min_lat, max_lon, max_lat = bounds
    transform = ((max_lon - min_lon) / w, 0.0, min_lon, 0.0, -(max_lat - min_lat) / h, max_lat)
    return img, oil, transform


def main():
    os.chdir(ROOT / "backend")
    from app.database import SessionLocal
    from app.models import Incident
    db = SessionLocal()
    cache = {}
    try:
        for inc in db.query(Incident).order_by(Incident.id).all():
            d = copy.deepcopy(inc.data or {})
            sp = d.get("spill") or {}
            if "centroid_lat" not in sp or not inc.result_dir or not d.get("bounds"):
                continue
            key = str(inc.result_dir)
            if key not in cache:
                cache[key] = load(inc.result_dir, d["bounds"])
            img, oil, transform = cache[key]
            check_spills(img, oil, transform, [sp])
            d["spill"] = sp
            inc.data = d
            r = sp.get("lookalike_check") or {}
            top = ", ".join(f"{e['name']} {'+' if e['effect'] > 0 else ''}{e['effect']}" for e in r.get("evidence", [])[:4])
            print(f"Incident #{inc.id}: {r.get('verdict')} (oil score {r.get('oil_score')})  [{top}]")
        db.commit()
    finally:
        db.close()
    print("Done. Refresh an incident page to see the 'Look-alike check' panel.")


if __name__ == "__main__":
    main()
