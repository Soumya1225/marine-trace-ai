"""Automatic satellite feed.

Every few hours (FEED_INTERVAL_HOURS) the backend asks the Copernicus Data Space catalogue which new Sentinel-1
radar images cover each watch zone that has "auto check" switched on. New images are listed on the Satellite
feed page. An analyst can download and analyse one with a click, or FEED_AUTO_ANALYSE=true does it automatically.
The catalogue search needs no login; the download needs a free Copernicus account (CDSE_USER / CDSE_PASSWORD).
"""
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app import config
from app.database import SessionLocal
from app.models import AISFile, AnalysisJob, SatellitePass, Scene, WatchZone

FEED_DIR_NAME = "auto_feed"
_lock = threading.Lock()


def _naive(text):
    if not text:
        return None
    t = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return t.astimezone(timezone.utc).replace(tzinfo=None)


def _footprint(product):
    try:
        return product["GeoFootprint"]["coordinates"][0]
    except (KeyError, IndexError, TypeError):
        return None


def check_zone(db, zone, days=None):
    """Search Copernicus for Sentinel-1 images over this zone. Returns the number of new images found."""
    from ml.preprocessing.download_sentinel1 import search
    days = days or config.FEED_LOOKBACK_DAYS
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    products = search((zone.min_lon, zone.min_lat, zone.max_lon, zone.max_lat),
                      start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    have = {p.name for p in db.query(SatellitePass).filter(SatellitePass.zone_id == zone.id)}
    new = 0
    for p in products:
        if p["Name"] in have:
            continue
        db.add(SatellitePass(zone_id=zone.id, product_id=p["Id"], name=p["Name"],
                             acquired_at=_naive(p.get("ContentDate", {}).get("Start")),
                             size_mb=round((p.get("ContentLength") or 0) / 1e6, 1), footprint=_footprint(p)))
        new += 1
    zone.last_checked_at = datetime.utcnow()
    db.commit()
    return new


def check_all(db, only_auto=True):
    q = db.query(WatchZone).filter(WatchZone.active.is_(True))
    if only_auto:
        q = q.filter(WatchZone.auto_check.is_(True))
    result = {}
    for z in q:
        try:
            result[z.name] = check_zone(db, z)
        except Exception as exc:                                  # one zone failing must not stop the others
            result[z.name] = f"error: {exc}"
    return result


def download_and_analyse(pass_id, user_id=None):
    """Background task: download the image, register it as a scene and run the normal analysis on the zone area."""
    from app.services.analysis_service import run_job
    from ml import config as mlc
    from ml.preprocessing.download_sentinel1 import download

    db = SessionLocal()
    sp = db.get(SatellitePass, pass_id)
    try:
        if not (os.getenv("CDSE_USER") and os.getenv("CDSE_PASSWORD")):
            raise RuntimeError("Add CDSE_USER and CDSE_PASSWORD (free Copernicus account) to the .env file to download images")
        sp.status, sp.message = "downloading", f"Downloading about {sp.size_mb:.0f} MB from Copernicus"
        db.commit()
        out_dir = mlc.RAW_DIR / "sentinel1" / FEED_DIR_NAME
        path = Path(download({"Id": sp.product_id, "Name": sp.name}, out_dir=str(out_dir)))
        z = sp.zone
        scene = Scene(filename=path.name, path=str(path), acquisition_time=sp.acquired_at,
                      bbox=[z.min_lon, z.min_lat, z.max_lon, z.max_lat],
                      size_mb=round(path.stat().st_size / 1e6, 2), uploaded_by=user_id)
        db.add(scene)
        db.commit()
        from app.services.ais_live import live_files_for
        ais_ids = []
        for f in live_files_for(sp.acquired_at or datetime.utcnow()):
            rec = AISFile(filename=f.name, path=str(f), size_mb=round(f.stat().st_size / 1e6, 2), uploaded_by=user_id)
            db.add(rec)
            db.flush()
            ais_ids.append(rec.id)
        db.commit()
        job = AnalysisJob(scene_id=scene.id, ais_ids=ais_ids, created_by=user_id,
                          params={"bbox": [z.min_lon, z.min_lat, z.max_lon, z.max_lat],
                                  "acquisition_time": sp.acquired_at.isoformat() if sp.acquired_at else None})
        db.add(job)
        db.commit()
        sp.scene_id, sp.job_id, sp.status, sp.message = scene.id, job.id, "analysing", f"Analysis #{job.id} running"
        db.commit()
        run_job(job.id)
        db.refresh(job)
        sp.status = "done" if job.state == "done" else "failed"
        sp.message = job.message if job.state == "done" else (job.error or job.message or "Analysis failed")[:390]
        db.commit()
    except Exception as exc:
        sp.status, sp.message = "failed", str(exc)[:390]
        db.commit()
    finally:
        db.close()


def _loop():
    time.sleep(30)                                                 # let the server finish starting
    while True:
        if _lock.acquire(blocking=False):
            db = SessionLocal()
            try:
                found = check_all(db, only_auto=True)
                if found:
                    print(f"[feed] checked {len(found)} zone(s): {found}")
                if config.FEED_AUTO_ANALYSE:
                    for sp in db.query(SatellitePass).filter(SatellitePass.status == "new").order_by(SatellitePass.acquired_at):
                        if sp.zone and sp.zone.auto_check:
                            download_and_analyse(sp.id)
            except Exception as exc:
                print(f"[feed] check failed: {exc}")
            finally:
                db.close()
                _lock.release()
        time.sleep(max(config.FEED_INTERVAL_HOURS, 0.25) * 3600)


def start_scheduler():
    if config.FEED_INTERVAL_HOURS > 0:
        threading.Thread(target=_loop, name="satellite-feed", daemon=True).start()
        print(f"[feed] automatic satellite check every {config.FEED_INTERVAL_HOURS:g} h for zones with auto check on")
