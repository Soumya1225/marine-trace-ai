"""Learn from analyst feedback (run after 'Export feedback' on the Model performance page).

1. Look-alike check: compares the rule-based verdict with the analyst labels, then fits a small logistic
   regression on the same physical features (contrast, edges, texture, wind, shape, model confidence) and
   reports its leave-one-out accuracy. With enough reviewed cases the learned weights can replace the
   literature thresholds.
2. Vessel ranking: how often the ship the analyst confirmed was ranked #1 / in the top 3.
3. Detection: lists the new radar crops (SOS format) ready for fine-tuning the detection models.

Results are written to models/metrics/feedback_retrain.json and shown on the Model performance page.

    python scripts/retrain_from_feedback.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.config import DATA_DIR, METRICS_DIR  # noqa: E402

FB = DATA_DIR / "feedback"
FEATURES = ["contrast", "edge_sharpness", "texture", "wind_ms", "elongation", "confidence"]
MIN_CASES = 6


def lookalike():
    p = FB / "lookalike.csv"
    if not p.exists() or p.stat().st_size == 0:
        return {"status": "no reviewed incidents yet"}
    df = pd.read_csv(p)
    y = (df["label"] == "oil").astype(int).to_numpy()
    out = {"reviewed": int(len(df)), "oil": int(y.sum()), "lookalike": int(len(y) - y.sum())}
    known = df["check_verdict"].isin(["Likely oil", "Likely look-alike"])
    if known.any():
        pred = (df.loc[known, "check_verdict"] == "Likely oil").astype(int).to_numpy()
        out["rule_check_accuracy"] = round(float((pred == y[known.to_numpy()]).mean()), 3)
    cols = [c for c in FEATURES if c in df.columns]
    X = df[cols].astype(float)
    X = X.fillna(X.median()).fillna(0).to_numpy()
    if len(df) < MIN_CASES or y.min() == y.max():
        out["learned_model"] = f"needs at least {MIN_CASES} reviewed incidents with both confirmed and rejected cases"
        return out
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    hits = 0
    for i in range(len(y)):                                           # leave-one-out
        m = np.ones(len(y), bool)
        m[i] = False
        if y[m].min() == y[m].max():
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500)).fit(X[m], y[m])
        hits += int(clf.predict(X[i:i + 1])[0] == y[i])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500)).fit(X, y)
    out["learned_accuracy_loo"] = round(hits / len(y), 3)
    out["learned_weights"] = {c: round(float(w), 3) for c, w in zip(cols, clf[-1].coef_[0])}
    return out


def attribution():
    p = FB / "attribution.csv"
    if not p.exists() or p.stat().st_size == 0:
        return {"status": "no reviewed suspects yet"}
    df = pd.read_csv(p)
    conf = df[df["label"] == 1]
    out = {"reviewed": int(len(df)), "confirmed": int(len(conf)), "dismissed": int((df["label"] == 0).sum())}
    if len(conf):
        out["confirmed_top1"] = round(float((conf["rank"] == 1).mean()), 3)
        out["confirmed_top3"] = round(float((conf["rank"] <= 3).mean()), 3)
    return out


def main():
    det = sorted((FB / "detection").glob("*_sat.jpg")) if (FB / "detection").exists() else []
    res = {"run_at": datetime.utcnow().isoformat(timespec="seconds") + "Z", "lookalike": lookalike(),
           "attribution": attribution(), "detection_pairs": len(det),
           "detection_note": "Radar crops with analyst masks in SOS format (Datasets/feedback/detection): "
                             "add them to the training set to fine-tune the detection models."}
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "feedback_retrain.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
