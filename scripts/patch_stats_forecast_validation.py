"""One-off patch: make the /metrics endpoint return models/metrics/forecast_validation.json."""
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "backend" / "app" / "routers" / "stats.py"
s = p.read_text(encoding="utf-8")
if "forecast_validation" in s:
    print("stats.py already patched")
else:
    a = '"lstm": read("lstm_drift.json"),'
    b = '"lstm": read("lstm_drift.json"), "forecast_validation": read("forecast_validation.json"),'
    c = '("summary", "attribution", "lstm_drift")'
    d = '("summary", "attribution", "lstm_drift", "forecast_validation")'
    assert s.count(a) == 1 and s.count(c) == 1, "stats.py looks different from expected - send it to Claude"
    p.write_text(s.replace(a, b).replace(c, d), encoding="utf-8")
    print("stats.py patched")
