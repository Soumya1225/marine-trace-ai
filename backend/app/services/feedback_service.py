"""Feedback loop: turn analyst decisions into new training data.

Analysts confirm or reject incidents and confirm or dismiss suspect ships. Those decisions are ground truth
the system did not have before. export_feedback() writes them to Datasets/feedback/:

  detection/<id>_sat.jpg + <id>_mask.png  radar crop of every reviewed spill with its oil mask
                                         (rejected = empty mask: a look-alike example). Same format as the SOS
                                         dataset, so the detection models can be fine-tuned on it.
  lookalike.csv                          look-alike check features + analyst label (oil / look-alike)
  attribution.csv                        ranker features of reviewed suspects + label (1 confirmed, 0 dismissed)
  summary.json                           counts and how well the system agreed with the analysts

feedback_summary() gives the live numbers for the Model performance page.
"""
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from app.models import Incident, Suspect
from ml.config import DATA_DIR, METRICS_DIR

OUT = DATA_DIR / "feedback"
CROP = 256
REVIEWED = ("confirmed", "rejected")
ATTR_FEATURES = ["min_dist_km", "alignment", "overlap", "hours_before", "max_gap_h", "speed_ratio", "loiter_h",
                 "type_risk", "length_m"]


def _label(status):
    return "oil" if status == "confirmed" else "look-alike"


def _counts(db):
    incs = db.query(Incident).all()
    reviewed = [i for i in incs if i.status in REVIEWED]
    agree = checked = 0
    for i in reviewed:
        v = (((i.data or {}).get("spill") or {}).get("lookalike_check") or {}).get("verdict")
        if v in ("Likely oil", "Likely look-alike"):
            checked += 1
            agree += (v == "Likely oil") == (i.status == "confirmed")
    sus = db.query(Suspect).filter(Suspect.review_status.in_(("confirmed", "dismissed"))).all()
    conf = [s for s in sus if s.review_status == "confirmed"]
    top1 = sum(1 for s in conf if s.rank == 1)
    return {
        "incidents_total": len(incs),
        "incidents_confirmed": sum(i.status == "confirmed" for i in incs),
        "incidents_rejected": sum(i.status == "rejected" for i in incs),
        "lookalike_check_agreement": round(agree / checked, 3) if checked else None,
        "lookalike_checked": checked,
        "suspects_confirmed": len(conf),
        "suspects_dismissed": len(sus) - len(conf),
        "confirmed_ranked_first": round(top1 / len(conf), 3) if conf else None,
    }


def feedback_summary(db):
    last = None
    p = OUT / "summary.json"
    if p.exists():
        last = json.loads(p.read_text()).get("exported_at")
    learned = METRICS_DIR / "feedback_retrain.json"
    return {**_counts(db), "last_export": last, "folder": str(OUT),
            "retrain": json.loads(learned.read_text()) if learned.exists() else None}


def _crop(inc):
    d = inc.data or {}
    sp, b = d.get("spill") or {}, d.get("bounds")
    base = Path(inc.result_dir or "")
    if not b or not (base / "sar.png").exists() or not (base / "mask.png").exists():
        return None
    img = np.asarray(Image.open(base / "sar.png").convert("L"))
    m = np.asarray(Image.open(base / "mask.png").convert("RGBA"))
    oil = (m[..., 0] > 200) & (m[..., 1] < 100) & (m[..., 3] > 150)
    h, w = img.shape
    r = int((b[3] - sp["centroid_lat"]) / (b[3] - b[1]) * h)
    c = int((sp["centroid_lon"] - b[0]) / (b[2] - b[0]) * w)
    r0, c0 = min(max(r - CROP // 2, 0), max(h - CROP, 0)), min(max(c - CROP // 2, 0), max(w - CROP, 0))
    img, oil = img[r0:r0 + CROP, c0:c0 + CROP], oil[r0:r0 + CROP, c0:c0 + CROP]
    if inc.status == "rejected":
        oil = np.zeros_like(oil)                      # analyst says: no oil here (look-alike)
    return img, oil


def export_feedback(db):
    (OUT / "detection").mkdir(parents=True, exist_ok=True)
    incs = db.query(Incident).filter(Incident.status.in_(REVIEWED)).order_by(Incident.id).all()
    n_img = 0
    la_rows = []
    for i in incs:
        cr = _crop(i)
        if cr is not None:
            img, oil = cr
            Image.fromarray(img).save(OUT / "detection" / f"incident{i.id}_sat.jpg", quality=95)
            Image.fromarray((oil * 255).astype(np.uint8)).save(OUT / "detection" / f"incident{i.id}_mask.png")
            n_img += 1
        sp = (i.data or {}).get("spill") or {}
        la = sp.get("lookalike_check") or {}
        la_rows.append({"incident_id": i.id, "label": _label(i.status), "check_verdict": la.get("verdict"),
                        "oil_score": la.get("oil_score"), **(la.get("features") or {}),
                        "wind_ms": sp.get("wind_speed_ms"), "elongation": sp.get("elongation"),
                        "area_km2": sp.get("area_km2"), "confidence": sp.get("confidence"),
                        "model_disagreement": sp.get("model_disagreement")})
    _write_csv(OUT / "lookalike.csv", la_rows)

    at_rows = []
    for s in db.query(Suspect).filter(Suspect.review_status.in_(("confirmed", "dismissed"))).all():
        f = s.features or {}
        at_rows.append({"incident_id": s.incident_id, "mmsi": s.mmsi, "vessel_name": s.vessel_name, "rank": s.rank,
                        "score": s.score, "label": int(s.review_status == "confirmed"),
                        **{k: f.get(k) for k in ATTR_FEATURES}})
    _write_csv(OUT / "attribution.csv", at_rows)

    summary = {**_counts(db), "exported_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
               "detection_pairs": n_img, "lookalike_rows": len(la_rows), "attribution_rows": len(at_rows),
               "next_step": "python scripts/retrain_from_feedback.py"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
