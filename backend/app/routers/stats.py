"""Dashboard stats, vessels, alerts, model metrics and activity log."""
import json
from collections import Counter
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import ANALYST, get_current_user
from app.database import get_db
from app.models import ActivityLog, Alert, Incident, Suspect, User, Vessel
from app.services.serializers import incident_summary, iso

router = APIRouter(prefix="/api", tags=["dashboard"])
OPEN = ("detected", "under_review", "confirmed")


@router.get("/stats")
def stats(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    incidents = db.query(Incident).all()
    now = datetime.utcnow()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    days = [(now - timedelta(days=i)).date() for i in range(29, -1, -1)]
    per_day = Counter(i.created_at.date() for i in incidents if i.created_at and i.created_at.date() >= days[0])
    by_image = Counter(i.detected_at.date() for i in incidents if i.detected_at)
    from ml.analytics.hotspots import hotspots, trend
    rows = []
    for i in incidents:
        sp = (i.data or {}).get("spill") or {}
        rows.append({"id": i.id, "lat": i.centroid_lat, "lon": i.centroid_lon, "area_km2": i.area_km2,
                     "priority": i.priority, "status": i.status,
                     "date": i.detected_at.date().isoformat() if i.detected_at else None,
                     "verdict": (sp.get("lookalike_check") or {}).get("verdict"),
                     "non_ship": bool((sp.get("infrastructure") or {}).get("possible_non_ship_source"))})
    try:
        from ml.impact.assessment import load_sites
        sites = [s for s in load_sites() if "lat" in s and "lon" in s and "name" in s]
    except Exception:
        sites = []
    return {
        "total_incidents": len(incidents),
        "open_incidents": sum(i.status in OPEN for i in incidents),
        "critical_open": sum(i.status in OPEN and i.priority == "Critical" for i in incidents),
        "area_this_month_km2": round(sum(i.area_km2 for i in incidents if i.detected_at and i.detected_at >= month_start), 3),
        "area_total_km2": round(sum(i.area_km2 for i in incidents), 3),
        "vessels_tracked": db.query(Vessel).count(),
        "confirmed_suspects": db.query(Suspect).filter(Suspect.review_status == "confirmed").count(),
        "unread_alerts": db.query(Alert).filter(Alert.read.is_(False)).count(),
        "by_status": dict(Counter(i.status for i in incidents)),
        "by_priority": dict(Counter(i.priority for i in incidents)),
        "per_day": [{"date": d.isoformat(), "count": per_day.get(d, 0)} for d in days],
        "by_image_date": [{"date": d.isoformat(), "count": n} for d, n in sorted(by_image.items())],
        "hotspots": hotspots(rows, sites),
        "trend": trend(rows),
        "by_verdict": dict(Counter(r["verdict"] or "Not checked" for r in rows if r["status"] != "rejected")),
        "possible_non_ship": sum(r["non_ship"] for r in rows),
        "recent": [incident_summary(i) for i in sorted(incidents, key=lambda x: x.id, reverse=True)[:8]],
        "map": [{"id": r["id"], "lat": r["lat"], "lon": r["lon"], "priority": r["priority"], "status": r["status"],
                 "area_km2": r["area_km2"], "verdict": r["verdict"]} for r in rows],
    }


@router.get("/vessels")
def vessels(q: str | None = None, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    query = db.query(Vessel)
    if q:
        query = query.filter((Vessel.name.ilike(f"%{q}%")) | (Vessel.mmsi.cast(str).ilike(f"%{q}%")))
    out = []
    for v in query.order_by(Vessel.last_seen.desc()).limit(300):
        sus = db.query(Suspect).filter(Suspect.mmsi == v.mmsi).all()
        out.append({"mmsi": v.mmsi, "name": v.name, "vessel_type": v.vessel_type, "length_m": v.length_m,
                    "last_seen": iso(v.last_seen), "incidents": len(sus),
                    "max_score": max((s.score for s in sus), default=0),
                    "confirmed": sum(s.review_status == "confirmed" for s in sus)})
    return out


@router.get("/vessels/{mmsi}")
def vessel_profile(mmsi: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    v = db.get(Vessel, mmsi)
    if not v:
        raise HTTPException(404, "Vessel not found")
    sus = db.query(Suspect).filter(Suspect.mmsi == mmsi).order_by(Suspect.id.desc()).all()
    history = []
    for s in sus:
        inc = db.get(Incident, s.incident_id)
        history.append({"incident_id": inc.id, "title": inc.title, "detected_at": iso(inc.detected_at),
                        "rank": s.rank, "score": s.score, "review_status": s.review_status,
                        "incident_status": inc.status, "priority": inc.priority})
    return {"mmsi": v.mmsi, "name": v.name, "imo": v.imo, "vessel_type": v.vessel_type, "length_m": v.length_m,
            "last_seen": iso(v.last_seen), "history": history,
            "latest_track": sus[0].track if sus else [], "latest_evidence": sus[0].evidence if sus else []}


@router.get("/alerts")
def alerts(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    return [{"id": a.id, "incident_id": a.incident_id, "level": a.level, "message": a.message, "read": a.read,
             "emailed": a.emailed, "created_at": iso(a.created_at)}
            for a in db.query(Alert).order_by(Alert.id.desc()).limit(200)]


@router.post("/alerts/{alert_id}/read")
def mark_read(alert_id: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    a = db.get(Alert, alert_id)
    if not a:
        raise HTTPException(404, "Alert not found")
    a.read = True
    db.commit()
    return {"ok": True}


@router.post("/alerts/read-all")
def mark_all(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    db.query(Alert).filter(Alert.read.is_(False)).update({"read": True})
    db.commit()
    return {"ok": True}


@router.get("/metrics")
def metrics(_: User = Depends(get_current_user)):
    from ml.config import METRICS_DIR, ROOT
    def read(name):
        p = METRICS_DIR / name
        return json.loads(p.read_text()) if p.exists() else None
    tests = {p.stem: json.loads(p.read_text()) for p in sorted(METRICS_DIR.glob("test_*.json"))} if METRICS_DIR.exists() else {}
    runs = {}
    if METRICS_DIR.exists():
        for p in METRICS_DIR.glob("*.json"):
            if p.stem not in ("summary", "attribution", "lstm_drift", "forecast_validation", "drifter_validation", "lookalike_med", "model_comparison") and not p.stem.startswith("test_"):
                r = json.loads(p.read_text())
                if "run_id" in r:
                    runs[r["run_id"]] = r.get("history", [])
    fig_dir = ROOT / "reports" / "figures"
    figures = sorted(p.name for p in fig_dir.glob("*.png")) if fig_dir.exists() else []
    return {"segmentation_runs": read("summary.json") or [], "test_results": tests, "histories": runs,
            "attribution": read("attribution.json"), "lstm": read("lstm_drift.json"), "forecast_validation": read("forecast_validation.json"),
            "drifter_validation": read("drifter_validation.json"), "lookalike": read("lookalike_med.json"),
            "comparison": read("model_comparison.json"),
            "figures": figures}


@router.get("/activity")
def activity(db: Session = Depends(get_db), _: User = Depends(ANALYST)):
    return [{"id": a.id, "user": a.user.name if a.user else "System", "action": a.action, "detail": a.detail,
             "created_at": iso(a.created_at)} for a in db.query(ActivityLog).order_by(ActivityLog.id.desc()).limit(300)]
