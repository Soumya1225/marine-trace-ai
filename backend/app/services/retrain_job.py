"""Run the feedback retraining (scripts/retrain_from_feedback.py) in the background from the dashboard.

Steps: export the analyst decisions as training data (feedback_service.export_feedback), then run the retraining
script in a separate Python process so the API stays responsive. Only one run at a time. The script writes its
results to models/metrics/feedback_retrain.json, which the Model performance page already shows.
"""
import subprocess
import sys
import threading
from collections import deque
from datetime import datetime, timezone

from ml import config as mlc

SCRIPT = mlc.ROOT / "scripts" / "retrain_from_feedback.py"
TIMEOUT_S = 60 * 60

_lock = threading.Lock()
STATE = {"running": False, "started_at": None, "finished_at": None, "ok": None, "message": None,
         "exported": None, "log": []}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def status():
    return {**STATE, "script_found": SCRIPT.exists(), "log": STATE["log"][-40:]}


def _run(started_by):
    tail = deque(maxlen=200)
    try:
        proc = subprocess.Popen([sys.executable, str(SCRIPT)], cwd=str(mlc.ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            tail.append(line.rstrip())
            STATE["log"] = list(tail)
        code = proc.wait(timeout=TIMEOUT_S)
        STATE["ok"] = code == 0
        STATE["message"] = "Retraining finished." if code == 0 else f"Retraining stopped with an error (exit code {code})."
    except Exception as exc:                       # noqa: BLE001 - report any failure on the page
        STATE["ok"] = False
        STATE["message"] = f"Could not run the retraining: {exc}"
    finally:
        STATE["running"] = False
        STATE["finished_at"] = _now()
        _log_activity(started_by)


def _log_activity(user_id):
    try:
        from app.database import SessionLocal
        from app.models import ActivityLog
        with SessionLocal() as db:
            db.add(ActivityLog(user_id=user_id, action="feedback_retrain",
                               detail=("finished" if STATE["ok"] else "failed") + f": {STATE['message']}"))
            db.commit()
    except Exception:                              # noqa: BLE001 - logging must never break the job
        pass


def start(db, user):
    """Export feedback, then start the retraining script. Returns the new status, or raises RuntimeError."""
    from app.services.feedback_service import export_feedback
    if not SCRIPT.exists():
        raise RuntimeError(f"Retraining script not found: scripts/{SCRIPT.name}")
    with _lock:
        if STATE["running"]:
            raise RuntimeError("Retraining is already running.")
        exported = export_feedback(db)
        STATE.update(running=True, started_at=_now(), finished_at=None, ok=None, message="Retraining…",
                     exported={k: exported.get(k) for k in ("detection_pairs", "lookalike_rows", "attribution_rows")},
                     log=[])
        threading.Thread(target=_run, args=(user.id,), daemon=True).start()
    return status()
