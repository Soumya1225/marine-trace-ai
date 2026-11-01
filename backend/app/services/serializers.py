"""Convert database rows to JSON-friendly dicts for the API."""


def iso(dt):
    return dt.isoformat() + "Z" if dt else None


def suspect_dict(s, with_track=True):
    d = {"id": s.id, "mmsi": s.mmsi, "vessel_name": s.vessel_name, "vessel_type": s.vessel_type,
         "rank": s.rank, "score": s.score, "rule_score": s.rule_score, "ml_score": s.ml_score,
         "confidence": s.confidence, "level": s.level, "features": s.features, "evidence": s.evidence,
         "review_status": s.review_status, "review_note": s.review_note}
    if with_track:
        d["track"] = s.track
    return d


def incident_summary(i):
    top = i.suspects[0] if i.suspects else None
    return {"id": i.id, "title": i.title, "status": i.status, "priority": i.priority,
            "priority_score": i.priority_score, "area_km2": i.area_km2, "centroid_lat": i.centroid_lat,
            "centroid_lon": i.centroid_lon, "confidence": i.confidence, "detected_at": iso(i.detected_at),
            "created_at": iso(i.created_at), "updated_at": iso(i.updated_at), "job_id": i.job_id,
            "top_suspect": {"mmsi": top.mmsi, "vessel_name": top.vessel_name, "score": top.score} if top else None,
            "zones": (i.data or {}).get("zones", []),
            **case_fields(i)}


OPEN_STATUSES = ("detected", "under_review", "confirmed")


def case_fields(i):
    from datetime import datetime
    a = getattr(i, "assignee", None)
    due = getattr(i, "due_at", None)
    return {"assigned_to": {"id": a.id, "name": a.name} if a else None, "due_at": iso(due),
            "closed_at": iso(getattr(i, "closed_at", None)),
            "overdue": bool(due and i.status in OPEN_STATUSES and datetime.utcnow() > due)}


def incident_detail(i, scene, file_base):
    from pathlib import Path
    d = incident_summary(i)
    d.update({
        "spill": i.data.get("spill"), "forecast": i.data.get("forecast"), "impact": i.data.get("impact"),
        "decision": i.data.get("decision"), "backtrack": i.data.get("backtrack"),
        "environment": i.data.get("environment"), "detector": i.data.get("detector"),
        "ships": i.data.get("ships"), "evidence": i.data.get("evidence"),
        "scene": {"id": scene.id, "filename": scene.filename, "bounds": i.data.get("bounds"),
                  "sar_url": f"{file_base}/sar.png", "mask_url": f"{file_base}/mask.png",
                  "uncertainty_url": f"{file_base}/uncertainty.png" if (Path(i.result_dir) / "uncertainty.png").exists() else None},
        "suspects": [suspect_dict(s) for s in i.suspects],
        "notes": [{"id": n.id, "text": n.text, "user": n.user.name if n.user else "", "created_at": iso(n.created_at)}
                  for n in i.notes],
    })
    return d
