"""Analysis jobs and incident case management."""
import csv
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.auth import ADMIN, ANALYST, get_current_user
from app.database import get_db
from app.models import ActivityLog, AISFile, Alert, AnalysisJob, Incident, Note, SatellitePass, Scene, Suspect, User
from app.schemas import AnalyzeIn, NoteIn, StatusUpdate, SuspectReview
from app.services.activity import log
from app.services.analysis_service import run_job
from app.services.report_service import build_report
from app.services.serializers import incident_detail, incident_summary, iso, suspect_dict

router = APIRouter(prefix="/api", tags=["analysis & incidents"])
STATUSES = ["detected", "under_review", "confirmed", "closed", "rejected"]


def job_dict(j, db):
    incidents = db.query(Incident.id).filter(Incident.job_id == j.id).all()
    return {"id": j.id, "scene_id": j.scene_id, "scene": j.scene.filename if j.scene else None, "state": j.state,
            "progress": j.progress, "message": j.message, "error": j.error, "summary": j.summary,
            "created_at": iso(j.created_at), "finished_at": iso(j.finished_at),
            "incident_ids": [i.id for i in incidents]}


@router.post("/analyses")
def start_analysis(body: AnalyzeIn, tasks: BackgroundTasks, db: Session = Depends(get_db), user: User = Depends(ANALYST)):
    scene = db.get(Scene, body.scene_id)
    if not scene:
        raise HTTPException(404, "Satellite image not found")
    for aid in body.ais_ids:
        if not db.get(AISFile, aid):
            raise HTTPException(404, f"AIS file {aid} not found")
    if not (body.acquisition_time or scene.acquisition_time):
        raise HTTPException(400, "Enter the image acquisition time (UTC); it is not in the file name")
    if scene.filename.lower().endswith((".png", ".jpg", ".jpeg")) and not (body.bbox or scene.bbox):
        raise HTTPException(400, "Enter the image area (bbox) for PNG/JPG images")
    params = json.loads(body.model_dump_json(exclude={"scene_id", "ais_ids"}))
    job = AnalysisJob(scene_id=scene.id, ais_ids=body.ais_ids, params=params, created_by=user.id)
    db.add(job)
    db.commit()
    tasks.add_task(run_job, job.id)
    log(db, user, "analysis_started", f"Job {job.id} on {scene.filename}")
    return job_dict(job, db)


@router.get("/analyses")
def list_jobs(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    return [job_dict(j, db) for j in db.query(AnalysisJob).order_by(AnalysisJob.id.desc()).limit(50)]


@router.get("/analyses/{job_id}")
def get_job(job_id: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    j = db.get(AnalysisJob, job_id)
    if not j:
        raise HTTPException(404, "Analysis not found")
    return job_dict(j, db)


@router.get("/incidents")
def list_incidents(status: str | None = None, priority: str | None = None, q: str | None = None,
                   date_from: str | None = None, date_to: str | None = None, mmsi: int | None = None,
                   db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    query = db.query(Incident)
    if status:
        query = query.filter(Incident.status == status)
    if priority:
        query = query.filter(Incident.priority == priority)
    if q:
        query = query.filter(Incident.title.ilike(f"%{q}%"))
    if date_from:
        query = query.filter(Incident.detected_at >= datetime.fromisoformat(date_from))
    if date_to:
        query = query.filter(Incident.detected_at <= datetime.fromisoformat(date_to + "T23:59:59"))
    if mmsi:
        query = query.join(Suspect).filter(Suspect.mmsi == mmsi)
    return [incident_summary(i) for i in query.order_by(Incident.id.desc()).limit(500)]


def _get(db, incident_id):
    inc = db.get(Incident, incident_id)
    if not inc:
        raise HTTPException(404, "Incident not found")
    return inc


@router.get("/incidents/{incident_id}")
def get_incident(incident_id: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    inc = _get(db, incident_id)
    return incident_detail(inc, db.get(Scene, inc.scene_id), f"/files/job_{inc.job_id}")


@router.patch("/incidents/{incident_id}/status")
def set_status(incident_id: int, body: StatusUpdate, db: Session = Depends(get_db), user: User = Depends(ANALYST)):
    if body.status not in STATUSES:
        raise HTTPException(400, f"Status must be one of {STATUSES}")
    inc = _get(db, incident_id)
    old = inc.status
    inc.status = body.status
    inc.closed_at = datetime.utcnow() if body.status in ("closed", "rejected") else None
    if body.note:
        db.add(Note(incident_id=inc.id, user_id=user.id, text=f"[{old} → {body.status}] {body.note}"))
    db.commit()
    log(db, user, "status_changed", f"Incident {inc.id}: {old} -> {body.status}")
    return incident_summary(inc)


@router.post("/incidents/{incident_id}/notes")
def add_note(incident_id: int, body: NoteIn, db: Session = Depends(get_db), user: User = Depends(ANALYST)):
    inc = _get(db, incident_id)
    n = Note(incident_id=inc.id, user_id=user.id, text=body.text)
    db.add(n)
    db.commit()
    log(db, user, "note_added", f"Incident {inc.id}")
    return {"id": n.id, "text": n.text, "user": user.name, "created_at": iso(n.created_at)}


@router.patch("/suspects/{suspect_id}")
def review_suspect(suspect_id: int, body: SuspectReview, db: Session = Depends(get_db), user: User = Depends(ANALYST)):
    s = db.get(Suspect, suspect_id)
    if not s:
        raise HTTPException(404, "Suspect not found")
    if body.review_status not in ("pending", "confirmed", "dismissed"):
        raise HTTPException(400, "review_status must be pending, confirmed or dismissed")
    s.review_status, s.review_note = body.review_status, body.review_note
    db.commit()
    log(db, user, "suspect_reviewed", f"Incident {s.incident_id}: {s.vessel_name} ({s.mmsi}) -> {body.review_status}")
    return suspect_dict(s, with_track=False)


@router.get("/incidents/{incident_id}/report")
def report(incident_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    inc = _get(db, incident_id)
    pdf = build_report(inc, db.get(Scene, inc.scene_id))
    log(db, user, "report_downloaded", f"Incident {inc.id}")
    return FileResponse(pdf, media_type="application/pdf", filename=f"incident_{inc.id}_report.pdf")


def _geojson_bytes(inc):
    d = inc.data
    feats = [{"type": "Feature", "properties": {"kind": "spill", "area_km2": inc.area_km2, "priority": inc.priority},
              "geometry": {"type": "Polygon", "coordinates": [d["spill"]["polygon"]]}}]
    for f in d.get("forecast") or []:
        if f["polygon"]:
            feats.append({"type": "Feature", "properties": {"kind": "forecast", "hours": f["hours"]},
                          "geometry": {"type": "Polygon", "coordinates": [f["polygon"]]}})
    for s in inc.suspects:
        if len(s.track) > 1:
            feats.append({"type": "Feature", "properties": {"kind": "vessel_track", "mmsi": s.mmsi,
                                                            "name": s.vessel_name, "rank": s.rank, "score": s.score},
                          "geometry": {"type": "LineString", "coordinates": [[p["lon"], p["lat"]] for p in s.track]}})
    return json.dumps({"type": "FeatureCollection", "features": feats}, indent=1).encode()


def _csv_bytes(inc):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["rank", "mmsi", "vessel_name", "vessel_type", "score", "rule_score", "ml_score", "min_dist_km",
                "hours_before", "alignment", "overlap", "ais_gap_flag", "slowdown_flag", "review_status"])
    for s in inc.suspects:
        f = s.features
        w.writerow([s.rank, s.mmsi, s.vessel_name, s.vessel_type, s.score, s.rule_score, s.ml_score,
                    f.get("min_dist_km"), f.get("hours_before"), f.get("alignment"), f.get("overlap"),
                    f.get("ais_gap_flag"), f.get("slowdown_flag"), s.review_status])
    return buf.getvalue().encode()


@router.get("/incidents/{incident_id}/export/geojson")
def export_geojson(incident_id: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    inc = _get(db, incident_id)
    return StreamingResponse(io.BytesIO(_geojson_bytes(inc)), media_type="application/geo+json",
                             headers={"Content-Disposition": f"attachment; filename=incident_{inc.id}.geojson"})


@router.get("/incidents/{incident_id}/export/suspects.csv")
def export_csv(incident_id: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    inc = _get(db, incident_id)
    return StreamingResponse(io.BytesIO(_csv_bytes(inc)), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename=incident_{inc.id}_suspects.csv"})


@router.get("/incidents/{incident_id}/export/evidence.zip")
def export_evidence(incident_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """One-click evidence package: report, images, map data, suspects, notes, audit history and SHA-256 manifest."""
    inc = _get(db, incident_id)
    scene = db.get(Scene, inc.scene_id)
    files = {}
    try:
        files["report.pdf"] = Path(build_report(inc, scene)).read_bytes()
    except Exception as exc:                                    # the package is still useful without the PDF
        files["report_error.txt"] = f"PDF report could not be built: {exc}".encode()
    rd = Path(inc.result_dir) if inc.result_dir else None
    for name in ("sar.png", "mask.png", "uncertainty.png"):
        if rd and (rd / name).exists():
            files[f"images/{name}"] = (rd / name).read_bytes()
    files["map/incident.geojson"] = _geojson_bytes(inc)
    files["suspects/suspects.csv"] = _csv_bytes(inc)
    detail = incident_detail(inc, scene, f"/files/job_{inc.job_id}")
    files["incident.json"] = json.dumps(detail, indent=1, default=str).encode()
    pat = re.compile(rf"\b[Ii]ncident #?{inc.id}\b")
    history = [a for a in db.query(ActivityLog).filter(ActivityLog.detail.like(f"%{inc.id}%"))
               .order_by(ActivityLog.created_at).all() if pat.search(a.detail or "")]
    files["audit_trail.csv"] = "time_utc,user,action,detail\n".encode() + "".join(
        f'{iso(a.created_at)},{(a.user.name if getattr(a, "user", None) else "")},{a.action},"{(a.detail or "").replace(chr(34), "")}"\n'
        for a in history).encode()
    lines = [f"SpillWatch / Marine Trace AI - evidence package for incident {inc.id}",
             f"Created {datetime.utcnow().isoformat(timespec='seconds')}Z by {user.name}",
             f"Title: {inc.title}", f"Priority: {inc.priority}   Status: {inc.status}", "",
             "Input files (fingerprints recorded when the analysis ran):"]
    for f in (inc.data.get("evidence") or {}).get("files", []):
        lines.append(f"  {f.get('role', '')}: {f.get('name', '')}  sha256={f.get('sha256', '')}")
    lines += ["", "Files in this package (SHA-256):"]
    lines += [f"  {hashlib.sha256(b).hexdigest()}  {n}" for n, b in sorted(files.items())]
    files["MANIFEST.txt"] = "\n".join(lines).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(f"incident_{inc.id}_evidence/{n}", b)
    buf.seek(0)
    log(db, user, "evidence_downloaded", f"Incident {inc.id}")
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename=incident_{inc.id}_evidence.zip"})


# ------------------------------------------------------------------ delete (admin): remove test runs or duplicates
def _delete_incident(db, inc):
    """Remove an incident with its suspects, notes and alerts. Result files on disk are kept."""
    db.query(Alert).filter(Alert.incident_id == inc.id).delete(synchronize_session=False)
    db.delete(inc)                                   # suspects and notes go with it (cascade)


@router.delete("/incidents/{incident_id}")
def delete_incident(incident_id: int, db: Session = Depends(get_db), user: User = Depends(ADMIN)):
    inc = db.get(Incident, incident_id)
    if not inc:
        raise HTTPException(404, "Incident not found")
    title = inc.title
    _delete_incident(db, inc)
    db.commit()
    log(db, user, "incident_deleted", f"Incident {incident_id}: {title}")
    return {"deleted": incident_id}


@router.delete("/analyses/{job_id}")
def delete_analysis(job_id: int, db: Session = Depends(get_db), user: User = Depends(ADMIN)):
    """Remove one analysis run and every incident it created (e.g. a test run or a repeated run)."""
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(404, "Analysis not found")
    if job.state in ("queued", "running"):
        raise HTTPException(409, "This analysis is still running")
    incs = db.query(Incident).filter(Incident.job_id == job_id).all()
    ids = [i.id for i in incs]
    for inc in incs:
        _delete_incident(db, inc)
    db.query(SatellitePass).filter(SatellitePass.job_id == job_id).update({"job_id": None}, synchronize_session=False)
    db.delete(job)
    db.commit()
    log(db, user, "analysis_deleted", f"Analysis {job_id} and incidents {ids}")
    return {"deleted_analysis": job_id, "deleted_incidents": ids}

