"""Live ships (AISStream.io) for the watch zones."""
from fastapi import APIRouter, Depends, HTTPException

from app.auth import ADMIN, get_current_user
from app.database import get_db
from app.models import User, WatchZone
from app.services.ais_live import LIVE

router = APIRouter(prefix="/api/live", tags=["live AIS"])


@router.get("/status")
def status(_: User = Depends(get_current_user)):
    return LIVE.status()


@router.get("/ships")
def ships(zone_id: int | None = None, minutes: int = 60, db=Depends(get_db), _: User = Depends(get_current_user)):
    bbox = None
    if zone_id:
        z = db.get(WatchZone, zone_id)
        if not z:
            raise HTTPException(404, "Zone not found")
        bbox = [z.min_lon, z.min_lat, z.max_lon, z.max_lat]
    rows = LIVE.snapshot(bbox=bbox, minutes=max(5, min(minutes, 360)))
    return {"status": LIVE.status(), "ships": rows,
            "counts": {"ships": len(rows), "moving": sum(1 for r in rows if not r["stopped"] and not r["silent"]),
                       "stopped": sum(1 for r in rows if r["stopped"] and not r["silent"]),
                       "silent": sum(1 for r in rows if r["silent"])}}


@router.post("/restart")
def restart(_: User = Depends(ADMIN)):
    """Reconnect, e.g. after adding a watch zone or the API key."""
    LIVE.restart()
    return LIVE.status()
