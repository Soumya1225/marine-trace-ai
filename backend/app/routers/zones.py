"""Watch zones: sea areas that are monitored. Spills inside a zone are tagged with its name."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth import ANALYST, get_current_user
from app.database import get_db
from app.models import Incident, User, WatchZone
from app.services.activity import log
from app.services.serializers import OPEN_STATUSES, iso

router = APIRouter(prefix="/api/zones", tags=["watch zones"])

# approximate boxes around busy Indian ports / sensitive coasts, plus the case-study area
PRESETS = [
    ("Mumbai coast (JNPT / Mumbai port)", [72.60, 18.80, 73.10, 19.30], "Busy port and offshore oil fields (Mumbai High further west)"),
    ("Gulf of Kutch (Kandla / Mundra / Vadinar)", [68.80, 22.30, 70.50, 23.00], "Oil terminals and single-point moorings; Marine National Park"),
    ("Chennai / Ennore", [80.20, 12.90, 80.50, 13.40], "Site of the 2017 Ennore oil spill"),
    ("Kochi coast", [76.00, 9.80, 76.35, 10.15], "Port and refinery; backwaters"),
    ("Visakhapatnam", [83.20, 17.50, 83.50, 17.80], "Port and refinery"),
    ("Paradip (Odisha)", [86.50, 20.10, 86.90, 20.40], "Oil port and refinery near Bhitarkanika mangroves"),
    ("Huntington Beach, California (case study)", [-118.25, 33.42, -117.85, 33.72], "Oct 2021 pipeline spill used to test the system"),
]


class ZoneIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    bbox: list[float] = Field(min_length=4, max_length=4)      # min_lon, min_lat, max_lon, max_lat
    notes: str = ""
    active: bool = True
    auto_check: bool = False


class ZonePatch(BaseModel):
    name: str | None = None
    notes: str | None = None
    active: bool | None = None
    auto_check: bool | None = None


def _check_bbox(b):
    lo1, la1, lo2, la2 = b
    if not (-180 <= lo1 < lo2 <= 180 and -90 <= la1 < la2 <= 90):
        raise HTTPException(400, "Area must be min_lon, min_lat, max_lon, max_lat with min < max")
    if (lo2 - lo1) * (la2 - la1) > 25:
        raise HTTPException(400, "Area is too large (keep it under about 5 x 5 degrees)")


def _refresh_live():
    try:
        from app.services.ais_live import LIVE
        LIVE.restart()
    except Exception:
        pass


def inside(z, lat, lon):
    return z.min_lat <= lat <= z.max_lat and z.min_lon <= lon <= z.max_lon


def zones_for(db, lat, lon):
    """Names of active zones that contain this point."""
    return [z.name for z in db.query(WatchZone).filter(WatchZone.active.is_(True)) if inside(z, lat, lon)]


def zone_dict(z, incidents=()):
    hits = [i for i in incidents if inside(z, i.centroid_lat, i.centroid_lon)]
    return {"id": z.id, "name": z.name, "bbox": [z.min_lon, z.min_lat, z.max_lon, z.max_lat], "notes": z.notes,
            "active": z.active, "auto_check": z.auto_check, "last_checked_at": iso(z.last_checked_at),
            "created_at": iso(z.created_at), "incidents": len(hits),
            "open_incidents": sum(1 for i in hits if i.status in OPEN_STATUSES),
            "critical": sum(1 for i in hits if i.priority == "Critical")}


@router.get("")
def list_zones(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    incs = db.query(Incident).all()
    return [zone_dict(z, incs) for z in db.query(WatchZone).order_by(WatchZone.name)]


@router.post("")
def create_zone(body: ZoneIn, db: Session = Depends(get_db), user: User = Depends(ANALYST)):
    _check_bbox(body.bbox)
    z = WatchZone(name=body.name.strip(), min_lon=body.bbox[0], min_lat=body.bbox[1], max_lon=body.bbox[2],
                  max_lat=body.bbox[3], notes=body.notes, active=body.active, auto_check=body.auto_check,
                  created_by=user.id)
    db.add(z)
    db.commit()
    log(db, user, "zone_created", z.name)
    _refresh_live()
    return zone_dict(z, db.query(Incident).all())


@router.post("/presets")
def add_presets(db: Session = Depends(get_db), user: User = Depends(ANALYST)):
    have = {z.name for z in db.query(WatchZone)}
    added = 0
    for name, b, notes in PRESETS:
        if name not in have:
            db.add(WatchZone(name=name, min_lon=b[0], min_lat=b[1], max_lon=b[2], max_lat=b[3], notes=notes,
                             created_by=user.id))
            added += 1
    db.commit()
    log(db, user, "zones_presets", f"{added} preset zones added")
    _refresh_live()
    return {"added": added}


@router.patch("/{zone_id}")
def update_zone(zone_id: int, body: ZonePatch, db: Session = Depends(get_db), user: User = Depends(ANALYST)):
    z = db.get(WatchZone, zone_id)
    if not z:
        raise HTTPException(404, "Zone not found")
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(z, k, v)
    db.commit()
    log(db, user, "zone_updated", z.name)
    return zone_dict(z, db.query(Incident).all())


@router.delete("/{zone_id}")
def delete_zone(zone_id: int, db: Session = Depends(get_db), user: User = Depends(ANALYST)):
    z = db.get(WatchZone, zone_id)
    if not z:
        raise HTTPException(404, "Zone not found")
    db.delete(z)
    db.commit()
    log(db, user, "zone_deleted", z.name)
    _refresh_live()
    return {"ok": True}
