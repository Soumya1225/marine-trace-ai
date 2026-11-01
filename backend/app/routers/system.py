"""System health: is everything the analysis needs in place, and how fresh is the data?"""
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app import config
from app.auth import ADMIN, ANALYST
from app.database import get_db
from app.models import AISFile, Alert, AnalysisJob, Incident, Scene, User
from app.services.activity import log
from app.services.alert_service import channels, notify
from app.services.serializers import iso
from ml import config as mlc

router = APIRouter(prefix="/api/system", tags=["system"])
STARTED = datetime.utcnow()

DATASETS = [
    ("Sentinel-1 radar images", "raw/sentinel1"),
    ("AIS ship tracks", "raw/ais"),
    ("ERA5 wind", "raw/weather"),
    ("Ocean currents", "raw/ocean_current"),
    ("Coastline and land", "reference"),
    ("Training images (SOS)", "raw/sos"),
    ("Synthetic attribution cases", "synthetic"),
]


def _folder(path):
    p = Path(path)
    if not p.exists():
        return {"exists": False, "files": 0, "size_mb": 0, "latest": None}
    files = [f for f in p.rglob("*") if f.is_file()]
    latest = max((f.stat().st_mtime for f in files), default=None)
    return {"exists": True, "files": len(files), "size_mb": round(sum(f.stat().st_size for f in files) / 1e6, 1),
            "latest": iso(datetime.utcfromtimestamp(latest)) if latest else None}


def _file(path):
    p = Path(path)
    if not p.exists():
        return {"name": p.name, "ok": False}
    st = p.stat()
    return {"name": p.name, "ok": True, "size_mb": round(st.st_size / 1e6, 1),
            "updated": iso(datetime.utcfromtimestamp(st.st_mtime))}


@router.get("/health")
def health(db: Session = Depends(get_db), _: User = Depends(ANALYST)):
    from app.services.analysis_service import get_detector
    det = get_detector()
    seg = []
    for pattern in mlc.ENSEMBLE_MODELS:
        seg += [_file(f) for f in sorted(mlc.MODELS_DIR.glob(pattern))]
    models = {
        "detector_mode": det.mode, "detector": det.arch,
        "files": seg + [_file(mlc.RANKER_PATH)] + [_file(f) for f in sorted(mlc.MODELS_DIR.glob("lstm*"))],
    }
    data = [{"name": n, "folder": f"Datasets/{rel}", **_folder(mlc.DATA_DIR / rel)} for n, rel in DATASETS]
    last_job = db.query(AnalysisJob).order_by(AnalysisJob.id.desc()).first()
    counts = {
        "users": db.query(func.count(User.id)).scalar(), "satellite_images": db.query(func.count(Scene.id)).scalar(),
        "ais_files": db.query(func.count(AISFile.id)).scalar(), "analyses": db.query(func.count(AnalysisJob.id)).scalar(),
        "incidents": db.query(func.count(Incident.id)).scalar(),
        "open_incidents": db.query(func.count(Incident.id)).filter(Incident.status.in_(["detected", "under_review"])).scalar(),
        "alerts": db.query(func.count(Alert.id)).scalar(),
    }
    disk = shutil.disk_usage(config.STORAGE_DIR)
    from app.services.ais_live import LIVE
    live = LIVE.status()
    from app.auth import verify_password
    adm = db.query(User).filter(User.email == config.ADMIN_EMAIL.lower()).first()
    default_pw = bool(adm and verify_password(config.ADMIN_PASSWORD, adm.password_hash))
    checks = [
        ("AI detector loaded", det.mode != "threshold", det.arch),
        ("Ship ranker model", mlc.RANKER_PATH.exists(), mlc.RANKER_PATH.name),
        ("Coastline / land data", mlc.LAND_PATH.exists(), mlc.LAND_PATH.name),
        ("Wind data", data[2]["files"] > 0, f"{data[2]['files']} files"),
        ("Current data", data[3]["files"] > 0, f"{data[3]['files']} files"),
        ("Alert channel (e-mail or Telegram)", any(channels().values()),
         ", ".join(k for k, v in channels().items() if v) or "not configured (alerts only on the dashboard)"),
        ("Live AIS stream (AISStream.io)", live["connected"],
         f"{live['ships']} ships, {live['messages']} messages" if live["connected"] else (live["error"] or "not connected")),
        ("Default admin password changed", not default_pw, "still the first-run password: change it under Users" if default_pw else "changed"),
        ("Secret key set in .env", not config.SECRET_KEY.startswith("change-this"), "default key in use" if config.SECRET_KEY.startswith("change-this") else "custom key"),
        ("Free disk space > 2 GB", disk.free > 2e9, f"{disk.free / 1e9:.1f} GB free"),
    ]
    ok = sum(c[1] for c in checks)
    return {
        "status": "ok" if ok == len(checks) else "warning",
        "summary": f"{ok} of {len(checks)} checks passed",
        "checks": [{"name": n, "ok": bool(v), "detail": d} for n, v, d in checks],
        "uptime_hours": round((datetime.utcnow() - STARTED).total_seconds() / 3600, 1),
        "server_time": iso(datetime.utcnow()),
        "last_analysis": {"id": last_job.id, "state": last_job.state, "message": last_job.message,
                          "at": iso(last_job.finished_at or last_job.created_at)} if last_job else None,
        "models": models, "datasets": data, "counts": counts, "alert_channels": channels(),
        "disk": {"free_gb": round(disk.free / 1e9, 1), "total_gb": round(disk.total / 1e9, 1)},
    }


@router.post("/test-alert")
def test_alert(db: Session = Depends(get_db), user: User = Depends(ADMIN)):
    sent = notify("Test", f"Test alert sent by {user.name}. If you can read this, alerts are working.")
    log(db, user, "test_alert", f"email={sent['email']} telegram={sent['telegram']}")
    return {"sent": sent, "configured": channels()}


@router.get("/public-stats")
def public_stats(db: Session = Depends(get_db)):
    """Public numbers for the landing page (no names, positions or vessel details)."""
    from app.models import SatellitePass, WatchZone
    from app.services.ais_live import LIVE
    return {
        "incidents": db.query(func.count(Incident.id)).scalar(),
        "area_km2": round(db.query(func.coalesce(func.sum(Incident.area_km2), 0)).scalar() or 0, 1),
        "zones": db.query(func.count(WatchZone.id)).filter(WatchZone.active.is_(True)).scalar(),
        "images_found": db.query(func.count(SatellitePass.id)).scalar(),
        "live_ships": LIVE.status()["ships"],
        "live": LIVE.status()["connected"],
    }
