"""Feedback loop: analyst labels are turned into accuracy figures and a learned look-alike model."""
import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("retrain", Path(__file__).resolve().parents[1] / "scripts" / "retrain_from_feedback.py")
retrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrain)


def test_lookalike_feedback(tmp_path, monkeypatch):
    rows = []
    for k in range(8):
        oil = k % 2 == 0
        rows.append({"incident_id": k, "label": "oil" if oil else "look-alike",
                     "check_verdict": "Likely oil" if oil or k == 7 else "Likely look-alike",
                     "contrast": 3.0 + k * 0.1 if oil else 0.5, "edge_sharpness": 2.0 if oil else 1.0,
                     "texture": 0.7, "wind_ms": 6.0 if oil else 1.0, "elongation": 4.0, "confidence": 0.8})
    pd.DataFrame(rows).to_csv(tmp_path / "lookalike.csv", index=False)
    monkeypatch.setattr(retrain, "FB", tmp_path)
    r = retrain.lookalike()
    assert r["reviewed"] == 8 and r["rule_check_accuracy"] == 0.875
    assert r["learned_accuracy_loo"] >= 0.75 and "contrast" in r["learned_weights"]


def test_attribution_feedback(tmp_path, monkeypatch):
    pd.DataFrame([{"rank": 1, "label": 1}, {"rank": 4, "label": 1}, {"rank": 2, "label": 0}]).to_csv(tmp_path / "attribution.csv", index=False)
    monkeypatch.setattr(retrain, "FB", tmp_path)
    r = retrain.attribution()
    assert r["confirmed"] == 2 and r["confirmed_top1"] == 0.5 and r["confirmed_top3"] == 0.5
