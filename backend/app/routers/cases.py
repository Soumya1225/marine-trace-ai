"""Case management: assign incidents to analysts with a deadline, case board, and a case timeline."""
import re
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import ANALYST, get_current_user
from app.database import get_db
from app.models import ActivityLog, Alert, Incident, Note, User
from app.services.activity import log
from app.services.serializers import OPEN_STATUSES, incident_summary, iso
from app.services.assistant import SUGGESTIONS, answer
from app.services.response_plan import plan
from app.services.summary_page import build as build_summary

router = APIRouter(prefix="/api", tags=["case management"])

# response time targets by priority (how long the assigned analyst has to resolve the case)
SLA_HOURS = {"Critical": 6, "High": 24, "Medium": 72, "Low": 168}


class AssignIn(BaseModel):
    user_id: int | None = None
    due_at: str | None = None        # ISO date-time; empty = automatic deadline from the priority


@router.get("/cases/assignees")
def assignees(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    rows = db.query(User).filter(User.active.is_(True), User.role.in_(["admin", "analyst"])).order_by(User.name)
    return [{"id": u.id, "name": u.name, "role": u.role} for u in rows]


@router.patch("/incidents/{incident_id}/assign")
def assign(incident_id: int, body: AssignIn, db: Session = Depends(get_db), user: User = Depends(ANALYST)):
    inc = db.get(Incident, incident_id)
    if not inc:
        raise HTTPException(404, "Incident not found")
    who = None
    if body.user_id is not None:
        who = db.get(User, body.user_id)
        if not who or not who.active or who.role not in ("admin", "analyst"):
            raise HTTPException(400, "Choose an active admin or analyst")
    inc.assigned_to = who.id if who else None
    if who is None:
        inc.due_at = None
    elif body.due_at:
        inc.due_at = datetime.fromisoformat(body.due_at.replace("Z", ""))
    else:
        inc.due_at = datetime.utcnow() + timedelta(hours=SLA_HOURS.get(inc.priority, 72))
    if who and inc.status == "detected":
        inc.status = "under_review"
    text = (f"Assigned to {who.name}, due {inc.due_at:%Y-%m-%d %H:%M} UTC" if who else "Unassigned")
    db.add(Note(incident_id=inc.id, user_id=user.id, text=f"[case] {text}"))
    db.commit()
    db.refresh(inc)
    log(db, user, "case_assigned", f"Incident {inc.id}: {text}")
    return incident_summary(inc)


@router.get("/cases")
def case_board(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    rows = db.query(Incident).order_by(Incident.id.desc()).limit(500).all()
    cases = [incident_summary(i) for i in rows]
    open_ = [c for c in cases if c["status"] in OPEN_STATUSES]
    return {
        "cases": cases,
        "me": user.id,
        "sla_hours": SLA_HOURS,
        "counts": {
            "open": len(open_),
            "unassigned": sum(1 for c in open_ if not c["assigned_to"]),
            "mine": sum(1 for c in open_ if c["assigned_to"] and c["assigned_to"]["id"] == user.id),
            "overdue": sum(1 for c in open_ if c["overdue"]),
            "closed": sum(1 for c in cases if c["status"] in ("closed", "rejected")),
        },
    }


@router.get("/incidents/{incident_id}/timeline")
def timeline(incident_id: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    inc = db.get(Incident, incident_id)
    if not inc:
        raise HTTPException(404, "Incident not found")
    ev = []
    if inc.detected_at:
        ev.append({"at": iso(inc.detected_at), "kind": "image", "who": "", "text": "Satellite image taken"})
    ev.append({"at": iso(inc.created_at), "kind": "detected", "who": "system",
               "text": f"Spill detected ({inc.area_km2:.2f} km²), priority {inc.priority}"})
    for a in db.query(Alert).filter(Alert.incident_id == inc.id):
        ev.append({"at": iso(a.created_at), "kind": "alert", "who": "system",
                   "text": f"{a.level} alert{' sent by e-mail / Telegram' if a.emailed else ' shown on the dashboard'}"})
    pat = re.compile(rf"\b[Ii]ncident #?{inc.id}\b")
    for a in db.query(ActivityLog).filter(ActivityLog.detail.like(f"%{inc.id}%")):
        if pat.search(a.detail or "") and a.action != "note_added":
            ev.append({"at": iso(a.created_at), "kind": a.action, "who": a.user.name if a.user else "",
                       "text": (a.detail or "").split(":", 1)[-1].strip() or a.action.replace("_", " ")})
    for n in inc.notes:
        if not n.text.startswith("[case]"):
            ev.append({"at": iso(n.created_at), "kind": "note", "who": n.user.name if n.user else "", "text": n.text})
    if inc.closed_at:
        ev.append({"at": iso(inc.closed_at), "kind": "closed", "who": "", "text": f"Case {inc.status}"})
    ev.sort(key=lambda e: e["at"] or "")
    return ev


class AskIn(BaseModel):
    question: str


@router.get("/assistant/suggestions")
def suggestions(_: User = Depends(get_current_user)):
    return SUGGESTIONS


@router.post("/incidents/{incident_id}/ask")
def ask(incident_id: int, body: AskIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    inc = db.get(Incident, incident_id)
    if not inc:
        raise HTTPException(404, "Incident not found")
    if not body.question.strip():
        raise HTTPException(400, "Type a question")
    res = answer(inc, body.question)
    log(db, user, "assistant_question", f"Incident {inc.id}: {body.question[:120]}")
    return res


@router.get("/incidents/{incident_id}/response")
def response_plan(incident_id: int, vessel_speed_kn: float | None = None, skimmers: int | None = None,
                  cost_per_m3_at_sea: float | None = None, cost_per_km_shoreline: float | None = None,
                  db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    inc = db.get(Incident, incident_id)
    if not inc:
        raise HTTPException(404, "Incident not found")
    return plan(inc, {"vessel_speed_kn": vessel_speed_kn, "skimmers": skimmers,
                      "cost_per_m3_at_sea": cost_per_m3_at_sea, "cost_per_km_shoreline": cost_per_km_shoreline})


@router.get("/incidents/{incident_id}/summary.html")
def summary_page(incident_id: int, lang: str = "en", db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    from fastapi.responses import HTMLResponse
    inc = db.get(Incident, incident_id)
    if not inc:
        raise HTTPException(404, "Incident not found")
    log(db, user, "summary_opened", f"Incident {inc.id} ({lang})")
    return HTMLResponse(build_summary(inc, "hi" if lang == "hi" else "en"))


@router.get("/incidents/{incident_id}/optical")
def optical(incident_id: int, days: int = 3, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    """Sentinel-2 optical images around the radar image, to confirm or reject the slick by eye (A6)."""
    from app.services.optical import for_incident
    inc = db.get(Incident, incident_id)
    if not inc:
        raise HTTPException(404, "Incident not found")
    try:
        return for_incident(inc, days=max(1, min(days, 10)))
    except Exception as exc:                                     # noqa: BLE001 - catalogue offline etc.
        raise HTTPException(502, f"Copernicus catalogue not reachable: {exc}")


@router.get("/optical/quicklook/{asset_id}")
def optical_quicklook(asset_id: str, _: User = Depends(get_current_user)):
    from fastapi.responses import Response
    from app.services.optical import quicklook
    try:
        return Response(quicklook(asset_id), media_type="image/jpeg",
                        headers={"Cache-Control": "private, max-age=86400"})
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:                                     # noqa: BLE001
        raise HTTPException(502, f"Preview not available: {exc}")


# ------------------------------------------------------------------ response solution (tier, method, intercept, checklist)
def _suspect_list(db, incident_id):
    from app.models import Suspect
    rows = db.query(Suspect).filter(Suspect.incident_id == incident_id).order_by(Suspect.rank).all()
    return [{"vessel_name": s.vessel_name, "mmsi": s.mmsi, "score": s.score} for s in rows]


@router.get("/incidents/{incident_id}/solution")
def incident_solution(incident_id: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    from app.services.solution import build
    inc = db.get(Incident, incident_id)
    if not inc:
        raise HTTPException(404, "Incident not found")
    return build(inc, _suspect_list(db, incident_id))


class TaskUpdate(BaseModel):
    status: str


@router.patch("/incidents/{incident_id}/solution/tasks/{task_id}")
def update_task(incident_id: int, task_id: str, body: TaskUpdate, db: Session = Depends(get_db), user: User = Depends(ANALYST)):
    from sqlalchemy.orm.attributes import flag_modified
    from app.services.solution import TASK_STATUSES, build
    if body.status not in TASK_STATUSES:
        raise HTTPException(400, f"Status must be one of {', '.join(TASK_STATUSES)}")
    inc = db.get(Incident, incident_id)
    if not inc:
        raise HTTPException(404, "Incident not found")
    data = dict(inc.data or {})
    saved = dict(data.get("response_checklist") or {})
    saved[task_id] = {"status": body.status, "by": user.name, "at": datetime.utcnow().isoformat(timespec="seconds") + "Z"}
    data["response_checklist"] = saved
    inc.data = data
    flag_modified(inc, "data")
    db.commit()
    log(db, user, "response_task", f"Incident {incident_id}: task {task_id} -> {body.status}")
    return build(inc, _suspect_list(db, incident_id))


# ------------------------------------------------------------------ response solution, phase 2
def _incident_or_404(db, incident_id):
    inc = db.get(Incident, incident_id)
    if not inc:
        raise HTTPException(404, "Incident not found")
    return inc


@router.get("/incidents/{incident_id}/solution/followup")
def solution_followup(incident_id: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    """Sentinel-1 images after the detection and the predicted next passes over the spill."""
    from app.services.solution_extra import followup
    inc = _incident_or_404(db, incident_id)
    try:
        return followup(inc)
    except Exception as exc:                                     # noqa: BLE001 - catalogue offline etc.
        raise HTTPException(502, f"Copernicus catalogue not reachable: {exc}")


@router.get("/incidents/{incident_id}/solution/extra")
def solution_extra(incident_id: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    """Coastal warning texts and clean-up progress."""
    from app.services.solution import build
    from app.services.solution_extra import cleanup, warnings
    inc = _incident_or_404(db, incident_id)
    sol = build(inc, _suspect_list(db, incident_id))
    return {"warnings": warnings(inc, sol), "cleanup": cleanup(inc, plan(inc))}


class CleanupEntry(BaseModel):
    date: str | None = None
    oil_m3: float = 0
    shore_km: float = 0
    cost: float = 0
    note: str | None = None


@router.post("/incidents/{incident_id}/solution/cleanup")
def add_cleanup(incident_id: int, body: CleanupEntry, db: Session = Depends(get_db), user: User = Depends(ANALYST)):
    from sqlalchemy.orm.attributes import flag_modified
    from app.services.solution_extra import cleanup
    inc = _incident_or_404(db, incident_id)
    if min(body.oil_m3, body.shore_km, body.cost) < 0:
        raise HTTPException(400, "Values cannot be negative")
    data = dict(inc.data or {})
    rows = list(data.get("cleanup_log") or [])
    rows.append({"date": body.date or datetime.utcnow().strftime("%Y-%m-%d"), "oil_m3": body.oil_m3, "shore_km": body.shore_km,
                 "cost": body.cost, "note": (body.note or "")[:300], "by": user.name})
    data["cleanup_log"] = rows
    inc.data = data
    flag_modified(inc, "data")
    db.commit()
    log(db, user, "cleanup_logged", f"Incident {incident_id}: {body.oil_m3} m3, {body.shore_km} km, cost {body.cost}")
    return cleanup(inc, plan(inc))


@router.delete("/incidents/{incident_id}/solution/cleanup/{index}")
def delete_cleanup(incident_id: int, index: int, db: Session = Depends(get_db), user: User = Depends(ANALYST)):
    from sqlalchemy.orm.attributes import flag_modified
    from app.services.solution_extra import cleanup
    inc = _incident_or_404(db, incident_id)
    data = dict(inc.data or {})
    rows = list(data.get("cleanup_log") or [])
    if not 0 <= index < len(rows):
        raise HTTPException(404, "Entry not found")
    rows.pop(index)
    data["cleanup_log"] = rows
    inc.data = data
    flag_modified(inc, "data")
    db.commit()
    log(db, user, "cleanup_deleted", f"Incident {incident_id}: entry {index}")
    return cleanup(inc, plan(inc))


@router.get("/incidents/{incident_id}/casefile.html")
def case_file(incident_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Printable case file for the authorities (evidence, suspect, samples, damage estimate)."""
    from fastapi.responses import HTMLResponse
    from app.models import Scene, Suspect
    from app.services.solution import build
    from app.services.solution_extra import casefile_html
    inc = _incident_or_404(db, incident_id)
    suspects = db.query(Suspect).filter(Suspect.incident_id == incident_id).order_by(Suspect.rank).all()
    sol = build(inc, _suspect_list(db, incident_id))
    log(db, user, "casefile_opened", f"Incident {incident_id}")
    return HTMLResponse(casefile_html(inc, db.get(Scene, inc.scene_id), suspects, sol, plan(inc), user.name))
