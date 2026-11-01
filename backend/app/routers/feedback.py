"""Feedback loop API: how the system agrees with analyst decisions, export of those decisions as training data,
and retraining the models from them."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import ADMIN, ANALYST, get_current_user
from app.database import get_db
from app.models import User
from app.services import retrain_job
from app.services.activity import log
from app.services.feedback_service import export_feedback, feedback_summary

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


@router.get("")
def summary(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    return feedback_summary(db)


@router.post("/export")
def export(db: Session = Depends(get_db), _: User = Depends(ANALYST)):
    return export_feedback(db)


@router.get("/retrain")
def retrain_status(_: User = Depends(get_current_user)):
    return retrain_job.status()


@router.post("/retrain")
def retrain(db: Session = Depends(get_db), user: User = Depends(ADMIN)):
    try:
        st = retrain_job.start(db, user)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    log(db, user, "feedback_retrain", "started: export + scripts/retrain_from_feedback.py")
    return st
