"""Tests for the "Retrain models from feedback" job (B7). A tiny stand-in script replaces the real retraining,
so this runs in a second without models or data:
    python -m pytest tests/test_retrain_job.py -q
"""
import sys
import time
from pathlib import Path
from types import SimpleNamespace as N

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services import feedback_service, retrain_job  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setitem(retrain_job.STATE, "running", False)
    monkeypatch.setattr(feedback_service, "export_feedback",
                        lambda db: {"detection_pairs": 2, "lookalike_rows": 3, "attribution_rows": 4})
    monkeypatch.setattr(retrain_job, "_log_activity", lambda user_id: None)


def _wait():
    for _ in range(100):
        if not retrain_job.STATE["running"]:
            return
        time.sleep(0.05)
    raise AssertionError("retraining job did not finish")


def test_retrain_runs_script_and_reports_success(tmp_path, monkeypatch):
    script = tmp_path / "retrain_from_feedback.py"
    script.write_text("print('look-alike model: 80% leave-one-out')\nprint('done')\n")
    monkeypatch.setattr(retrain_job, "SCRIPT", script)
    st = retrain_job.start(None, N(id=1))
    assert st["running"] and st["exported"] == {"detection_pairs": 2, "lookalike_rows": 3, "attribution_rows": 4}
    _wait()
    st = retrain_job.status()
    assert st["ok"] is True and st["finished_at"]
    assert "done" in st["log"]


def test_retrain_reports_script_failure(tmp_path, monkeypatch):
    script = tmp_path / "retrain_from_feedback.py"
    script.write_text("import sys\nprint('not enough reviewed cases')\nsys.exit(2)\n")
    monkeypatch.setattr(retrain_job, "SCRIPT", script)
    retrain_job.start(None, N(id=1))
    _wait()
    st = retrain_job.status()
    assert st["ok"] is False and "exit code 2" in st["message"]


def test_retrain_refuses_second_run_and_missing_script(tmp_path, monkeypatch):
    monkeypatch.setattr(retrain_job, "SCRIPT", tmp_path / "missing.py")
    with pytest.raises(RuntimeError, match="not found"):
        retrain_job.start(None, N(id=1))
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(1)\n")
    monkeypatch.setattr(retrain_job, "SCRIPT", script)
    retrain_job.start(None, N(id=1))
    with pytest.raises(RuntimeError, match="already running"):
        retrain_job.start(None, N(id=1))
    _wait()
