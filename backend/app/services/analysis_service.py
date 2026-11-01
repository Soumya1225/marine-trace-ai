"""Runs the ML pipeline for an analysis job (in a background thread) and stores the results."""
import hashlib
import traceback
from datetime import datetime, timezone
from functools import lru_cache

from app.config import STORAGE_DIR
from app.database import SessionLocal
from app.models import AISFile, AnalysisJob, Incident, Suspect, Vessel
from app.services.activity import log
from app.services.alert_service import create_alert
from app.routers.zones import zones_for


@lru_cache(maxsize=1)
def get_detector():
    from ml.segmentation.predict import SpillDetector
    return SpillDetector()


def reset_models():
    """Call after training a new model so the next job reloads it."""
    get_detector.cache_clear()


def _naive_utc(value):
    if not value:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _upsert_vessel(db, s):
    v = db.get(Vessel, s["mmsi"])
    if v is None:
        v = Vessel(mmsi=s["mmsi"])
        db.add(v)
    v.name = s["vessel_name"]
    v.imo = s.get("imo")
    v.vessel_type = s["vessel_type"]
    v.length_m = s.get("length_m") or v.length_m
    seen = _naive_utc(s.get("closest_time"))
    if seen and (v.last_seen is None or seen > v.last_seen):
        v.last_seen = seen


def sha256_file(path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def run_job(job_id: int):
    from ml.attribution.scorer import load_ranker
    from ml.pipeline import analyze

    db = SessionLocal()
    job = db.get(AnalysisJob, job_id)
    try:
        job.state, job.progress, job.message = "running", 1, "Starting"
        db.commit()
        scene = job.scene
        ais_paths = [f.path for f in (db.get(AISFile, i) for i in job.ais_ids) if f]
        params = job.params or {}
        env_cfg = {k: params[k] for k in ("wind_u", "wind_v", "current_u", "current_v") if params.get(k) is not None}
        bbox = params.get("bbox") or scene.bbox
        acq = params.get("acquisition_time") or (scene.acquisition_time.isoformat() + "+00:00" if scene.acquisition_time else None)
        out_dir = STORAGE_DIR / "results" / f"job_{job.id}"

        def progress(pct, msg):
            job.progress, job.message = int(pct), msg
            db.commit()

        result = analyze(scene.path, ais_paths, tuple(bbox) if bbox else None, acq, env_cfg, out_dir,
                         detector=get_detector(), ranker=load_ranker(), progress=progress)
        progress(97, "Computing evidence fingerprints")
        evidence = {"analysed_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "detector": result["detector"]["model"],
                    "files": [{"role": "satellite image", "name": scene.filename, "sha256": sha256_file(scene.path)}]
                    + [{"role": "AIS data", "name": f.filename, "sha256": sha256_file(f.path)}
                       for f in (db.get(AISFile, i) for i in job.ais_ids) if f]}

        for r in result["spills"]:
            spill, dec = r["spill"], r["decision"]
            inc = Incident(
                job_id=job.id, scene_id=scene.id, spill_no=spill["spill_no"],
                title=f"Spill {spill['spill_no']} near {spill['centroid_lat']:.3f}, {spill['centroid_lon']:.3f}",
                priority=dec["priority"], priority_score=dec["priority_score"], area_km2=spill["area_km2"],
                centroid_lat=spill["centroid_lat"], centroid_lon=spill["centroid_lon"],
                confidence=spill.get("confidence"), detected_at=_naive_utc(spill.get("acquisition_time")),
                result_dir=str(out_dir),
                data={"spill": spill, "forecast": r["forecast"], "impact": r["impact"], "decision": dec,
                      "backtrack": r["backtrack"], "bounds": result["scene"]["bounds"],
                      "environment": result["environment"], "detector": result["detector"],
                      "ships": result.get("ships"), "evidence": evidence,
                      "zones": zones_for(db, spill["centroid_lat"], spill["centroid_lon"])},
            )
            db.add(inc)
            db.flush()
            for s in r["suspects"]:
                features = {k: v for k, v in s.items() if k not in ("track", "evidence")}
                db.add(Suspect(incident_id=inc.id, mmsi=s["mmsi"], vessel_name=s["vessel_name"],
                               vessel_type=s["vessel_type"], rank=s["rank"], score=s["score"],
                               rule_score=s["rule_score"], ml_score=s.get("ml_score"),
                               confidence=s["confidence"], level=s["level"], features=features,
                               evidence=s["evidence"], track=s.get("track", [])))
                _upsert_vessel(db, s)
            db.commit()
            if dec["priority"] in ("Critical", "High", "Medium"):
                zone_tag = f"[Watch zone: {', '.join(inc.data['zones'])}] " if inc.data.get("zones") else ""
                create_alert(db, inc, dec["priority"], zone_tag + dec["alert_message"])

        job.state, job.progress, job.message = "done", 100, f"Finished: {len(result['spills'])} spill(s) found"
        job.summary = {"spills": len(result["spills"]), "ais_vessels": result["ais"]["vessels"],
                       "ais_points": result["ais"]["points"], "detector": result["detector"]["model"],
                       "lookalike_pixels": result["lookalike_pixels"]}
        job.finished_at = datetime.utcnow()
        db.commit()
        log(db, None, "analysis_done", f"Job {job.id}: {len(result['spills'])} spill(s)")
    except Exception as exc:
        db.rollback()
        job = db.get(AnalysisJob, job_id)
        job.state, job.message = "failed", f"Analysis failed: {exc}"
        job.error = traceback.format_exc()
        job.finished_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()
