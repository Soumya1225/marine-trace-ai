"""Satellite feed API: new Sentinel-1 images over the watch zones."""
import os

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app import config
from app.auth import ANALYST, get_current_user
from app.database import get_db
from app.models import SatellitePass, User, WatchZone
from app.services.activity import log
from app.services.satellite_feed import check_zone, download_and_analyse
from app.services.serializers import iso

router = APIRouter(prefix="/api/feed", tags=["satellite feed"])


class CheckIn(BaseModel):
    zone_id: int | None = None
    days: int = 3


def pass_dict(p):
    return {"id": p.id, "zone_id": p.zone_id, "zone": p.zone.name if p.zone else "", "name": p.name,
            "acquired_at": iso(p.acquired_at), "size_mb": p.size_mb, "footprint": p.footprint, "status": p.status,
            "message": p.message, "scene_id": p.scene_id, "job_id": p.job_id, "found_at": iso(p.found_at)}


@router.get("")
def list_passes(zone_id: int | None = None, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    q = db.query(SatellitePass)
    if zone_id:
        q = q.filter(SatellitePass.zone_id == zone_id)
    return {
        "passes": [pass_dict(p) for p in q.order_by(SatellitePass.acquired_at.desc()).limit(300)],
        "settings": {"interval_hours": config.FEED_INTERVAL_HOURS, "lookback_days": config.FEED_LOOKBACK_DAYS,
                     "auto_analyse": config.FEED_AUTO_ANALYSE,
                     "can_download": bool(os.getenv("CDSE_USER") and os.getenv("CDSE_PASSWORD"))},
    }


@router.post("/check")
def check_now(body: CheckIn, db: Session = Depends(get_db), user: User = Depends(ANALYST)):
    """Search Copernicus right now (one zone, or every active zone)."""
    if not 1 <= body.days <= 30:
        raise HTTPException(400, "days must be between 1 and 30")
    zones = [db.get(WatchZone, body.zone_id)] if body.zone_id else db.query(WatchZone).filter(WatchZone.active.is_(True)).all()
    zones = [z for z in zones if z]
    if not zones:
        raise HTTPException(400, "Create a watch zone first")
    found, errors = {}, {}
    for z in zones:
        try:
            found[z.name] = check_zone(db, z, days=body.days)
        except Exception as exc:
            errors[z.name] = str(exc)[:200]
    log(db, user, "feed_checked", f"{sum(found.values())} new image(s) in {len(found)} zone(s)")
    return {"found": found, "errors": errors}


@router.post("/{pass_id}/analyse")
def analyse_pass(pass_id: int, tasks: BackgroundTasks, db: Session = Depends(get_db), user: User = Depends(ANALYST)):
    sp = db.get(SatellitePass, pass_id)
    if not sp:
        raise HTTPException(404, "Image not found")
    if sp.status in ("downloading", "analysing"):
        raise HTTPException(400, "Already in progress")
    sp.status, sp.message = "queued", "Waiting to download"
    db.commit()
    tasks.add_task(download_and_analyse, sp.id, user.id)
    log(db, user, "feed_analyse", sp.name)
    return pass_dict(sp)


@router.post("/{pass_id}/skip")
def skip_pass(pass_id: int, db: Session = Depends(get_db), user: User = Depends(ANALYST)):
    sp = db.get(SatellitePass, pass_id)
    if not sp:
        raise HTTPException(404, "Image not found")
    sp.status, sp.message = "skipped", f"Skipped by {user.name}"
    db.commit()
    return pass_dict(sp)
