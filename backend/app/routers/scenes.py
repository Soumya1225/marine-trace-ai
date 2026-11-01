"""Satellite scenes and AIS files: upload, register local files, list, demo data."""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.auth import ANALYST, get_current_user
from app.config import STORAGE_DIR
from app.database import get_db
from app.models import AISFile, Scene, User
from app.schemas import RegisterPath
from app.services.activity import log
from app.services.serializers import iso

router = APIRouter(prefix="/api", tags=["data"])
SCENE_EXT = {".zip", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _naive(dt):
    if dt and dt.tzinfo:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _scene_time(path, given):
    if given:
        return _naive(given)
    from ml.preprocessing.sar_preprocess import parse_acquisition_time
    t = parse_acquisition_time(Path(path).name)
    return _naive(t) if t else None


def scene_dict(s):
    return {"id": s.id, "filename": s.filename, "acquisition_time": iso(s.acquisition_time), "bbox": s.bbox,
            "size_mb": s.size_mb, "created_at": iso(s.created_at)}


def ais_dict(a):
    return {"id": a.id, "filename": a.filename, "size_mb": a.size_mb, "created_at": iso(a.created_at)}


def _save_upload(file: UploadFile, folder: str) -> Path:
    dest = STORAGE_DIR / folder / f"{datetime.utcnow():%Y%m%d%H%M%S}_{Path(file.filename).name}"
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out, length=8 * 1024 * 1024)
    return dest


def _existing_copy(db, model, filename, dest):
    """Same file uploaded again (same name and size): drop the new copy and reuse the earlier entry."""
    size = round(dest.stat().st_size / 1e6, 2)
    for old in db.query(model).filter(model.filename == filename).all():
        if old.size_mb == size and Path(old.path).exists():
            dest.unlink(missing_ok=True)
            return old
    return None


def _used_by(db, scene_id=None, ais_id=None):
    """Analyses / incidents that still use this file (found by column name, so no model import is needed)."""
    from app.database import Base
    users = []
    for mapper in Base.registry.mappers:
        cls = mapper.class_
        name = cls.__name__
        if scene_id is not None and hasattr(cls, "scene_id"):
            users += [f"{name} #{r.id}" for r in db.query(cls).filter(cls.scene_id == scene_id).limit(5)]
        if ais_id is not None and hasattr(cls, "ais_ids"):
            users += [f"{name} #{r.id}" for r in db.query(cls).all() if ais_id in (r.ais_ids or [])][:5]
    return users


def _delete_file(path):
    """Delete only copies uploaded through the website, never the original data files on this computer."""
    p = Path(path).resolve()
    if STORAGE_DIR.resolve() in p.parents and p.exists():
        p.unlink()
        return True
    return False


def _parse_bbox(text):
    if not text:
        return None
    vals = [float(v) for v in text.replace(" ", "").split(",")]
    if len(vals) != 4:
        raise HTTPException(400, "bbox must be: min_lon,min_lat,max_lon,max_lat")
    return vals


@router.get("/scenes")
def list_scenes(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    return [scene_dict(s) for s in db.query(Scene).order_by(Scene.id.desc())]


@router.post("/scenes")
def upload_scene(file: UploadFile = File(...), acquisition_time: str | None = Form(None),
                 bbox: str | None = Form(None), db: Session = Depends(get_db), user: User = Depends(ANALYST)):
    if Path(file.filename).suffix.lower() not in SCENE_EXT:
        raise HTTPException(400, "Upload a Sentinel-1 .zip, GeoTIFF, PNG or JPG file")
    dest = _save_upload(file, "scenes")
    old = _existing_copy(db, Scene, file.filename, dest)
    if old:
        return scene_dict(old)
    when = datetime.fromisoformat(acquisition_time.replace("Z", "+00:00")) if acquisition_time else None
    s = Scene(filename=file.filename, path=str(dest), acquisition_time=_scene_time(dest, when),
              bbox=_parse_bbox(bbox), size_mb=round(dest.stat().st_size / 1e6, 2), uploaded_by=user.id)
    db.add(s)
    db.commit()
    log(db, user, "scene_uploaded", file.filename)
    return scene_dict(s)


@router.post("/scenes/register")
def register_scene(body: RegisterPath, db: Session = Depends(get_db), user: User = Depends(ANALYST)):
    """Use a large file already on this computer (e.g. Datasets/raw/sentinel1/...zip) without uploading it."""
    p = Path(body.path).expanduser()
    if not p.is_absolute():
        p = (STORAGE_DIR.parents[1] / p).resolve()
    if not p.exists():
        raise HTTPException(400, f"File not found: {p}")
    s = Scene(filename=p.name, path=str(p), acquisition_time=_scene_time(p, body.acquisition_time),
              bbox=body.bbox, size_mb=round(p.stat().st_size / 1e6, 2), uploaded_by=user.id)
    db.add(s)
    db.commit()
    log(db, user, "scene_registered", p.name)
    return scene_dict(s)


@router.get("/ais")
def list_ais(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    return [ais_dict(a) for a in db.query(AISFile).order_by(AISFile.id.desc())]


@router.post("/ais")
def upload_ais(file: UploadFile = File(...), db: Session = Depends(get_db), user: User = Depends(ANALYST)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload AIS data as a .csv file")
    dest = _save_upload(file, "ais")
    old = _existing_copy(db, AISFile, file.filename, dest)
    if old:
        return ais_dict(old)
    a = AISFile(filename=file.filename, path=str(dest), size_mb=round(dest.stat().st_size / 1e6, 2), uploaded_by=user.id)
    db.add(a)
    db.commit()
    log(db, user, "ais_uploaded", file.filename)
    return ais_dict(a)


@router.post("/ais/register")
def register_ais(body: RegisterPath, db: Session = Depends(get_db), user: User = Depends(ANALYST)):
    p = Path(body.path).expanduser()
    if not p.is_absolute():
        p = (STORAGE_DIR.parents[1] / p).resolve()
    if not p.exists():
        raise HTTPException(400, f"File not found: {p}")
    a = AISFile(filename=p.name, path=str(p), size_mb=round(p.stat().st_size / 1e6, 2), uploaded_by=user.id)
    db.add(a)
    db.commit()
    log(db, user, "ais_registered", p.name)
    return ais_dict(a)


@router.delete("/scenes/{scene_id}")
def delete_scene(scene_id: int, db: Session = Depends(get_db), user: User = Depends(ANALYST)):
    s = db.get(Scene, scene_id)
    if not s:
        raise HTTPException(404, "Image not found")
    users = _used_by(db, scene_id=scene_id)
    if users:
        raise HTTPException(409, f"Cannot remove: still used by {', '.join(users)}. Its results would break.")
    removed = _delete_file(s.path)
    db.delete(s)
    db.commit()
    log(db, user, "scene_removed", s.filename)
    return {"ok": True, "file_deleted": removed}


@router.delete("/ais/{ais_id}")
def delete_ais(ais_id: int, db: Session = Depends(get_db), user: User = Depends(ANALYST)):
    a = db.get(AISFile, ais_id)
    if not a:
        raise HTTPException(404, "AIS file not found")
    users = _used_by(db, ais_id=ais_id)
    if users:
        raise HTTPException(409, f"Cannot remove: still used by {', '.join(users)}. Its results would break.")
    removed = _delete_file(a.path)
    db.delete(a)
    db.commit()
    log(db, user, "ais_removed", a.filename)
    return {"ok": True, "file_deleted": removed}


@router.post("/demo/setup")
def demo_setup(db: Session = Depends(get_db), user: User = Depends(ANALYST)):
    """Create synthetic demo data (Huntington Beach area) and register it, ready to analyse."""
    from ml.utils.demo_data import main as make_demo
    out = STORAGE_DIR / "demo"
    make_demo(out)
    meta = json.loads((out / "demo_meta.json").read_text())
    when = datetime.fromisoformat(meta["acquisition_time"])
    s = Scene(filename="demo_sar.png (synthetic)", path=str(out / "demo_sar.png"), acquisition_time=_naive(when),
              bbox=list(meta["bbox"]), size_mb=round((out / "demo_sar.png").stat().st_size / 1e6, 2), uploaded_by=user.id)
    a = AISFile(filename="demo_ais.csv (synthetic)", path=str(out / "demo_ais.csv"),
                size_mb=round((out / "demo_ais.csv").stat().st_size / 1e6, 2), uploaded_by=user.id)
    db.add_all([s, a])
    db.commit()
    log(db, user, "demo_created")
    return {"scene": scene_dict(s), "ais": ais_dict(a)}
