"""Train the ML ranker on synthetic cases and compare it with the rule-based score.

    python -m ml.attribution.train_ranker --cases Datasets/synthetic/cases.pkl
Saves models/ranker.pkl and models/metrics/attribution.json (Top-1 / Top-3 accuracy).
"""
import argparse
import json
import pickle

import numpy as np
import pandas as pd

from ml.attribution.attribute import attribute_spill
from ml.attribution.drift import Environment
from ml.attribution.scorer import FEATURE_NAMES, feature_vector
from ml.config import METRICS_DIR, RANKER_PATH, SYNTHETIC_DIR


def build_table(cases, ais, env_noise=0.2, seed=0):
    """Run attribution (rule-based) on every case and collect candidate features + labels."""
    rng = np.random.default_rng(seed)
    rows = []
    for cid, case in enumerate(cases):
        # forecast data is never perfect: perturb the environment the analyst would use
        env_cfg = {k: v * (1 + rng.normal(0, env_noise)) for k, v in case["env"].items()}
        suspects, _ = attribute_spill(case["spill"], ais, case["image_time"], Environment(**env_cfg),
                                      ranker=None, top_k=50, include_tracks=False)
        sp = case["spill"]
        ctx = {"ctx_area_km2": sp.get("area_km2", 0), "ctx_elongation": sp.get("elongation", 1),
               "ctx_length_km": sp.get("length_km", 0),
               "ctx_wind_ms": float(np.hypot(env_cfg["wind_u"], env_cfg["wind_v"])),
               "ctx_current_ms": float(np.hypot(env_cfg["current_u"], env_cfg["current_v"]))}
        for s in suspects:
            rows.append({"case": cid, "mmsi": s["mmsi"], "label": int(s["mmsi"] == case["true_mmsi"]),
                         "rule_score": s["rule_score"], **{k: s.get(k) for k in FEATURE_NAMES}, **ctx})
    return pd.DataFrame(rows)


def topk_accuracy(df, score_col, n_cases, k):
    hits = 0
    for _, g in df.groupby("case"):
        top = g.sort_values(score_col, ascending=False).head(k)
        hits += int(top.label.max() == 1)
    return round(hits / max(n_cases, 1), 4)


def make_model():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                             eval_metric="logloss"), "XGBoost"
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=300, max_depth=3), "GradientBoosting (sklearn)"


def train_lgbm_ranker(train, test):
    """Learning-to-rank: LightGBM LambdaRank optimises the ORDER of ships within each case (NDCG), instead of
    classifying each ship on its own like XGBoost. Each synthetic case is one query group."""
    from lightgbm import LGBMRanker
    train = train.sort_values("case")
    groups = train.groupby("case", sort=True).size().values
    model = LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=15,
                       min_child_samples=5, verbose=-1)
    model.fit(np.array([feature_vector(r) for r in train.to_dict("records")]), train.label.values, group=groups)
    with open(RANKER_PATH.parent / "lgbm_ranker.pkl", "wb") as fh:
        pickle.dump(model, fh)
    return model.predict(np.array([feature_vector(r) for r in test.to_dict("records")]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(SYNTHETIC_DIR / "cases.pkl"))
    ap.add_argument("--test-frac", type=float, default=0.25)
    args = ap.parse_args()
    with open(args.cases, "rb") as fh:
        bundle = pickle.load(fh)
    cases = bundle["cases"]
    ais = pd.read_pickle(bundle["ais_path"])

    table = build_table(cases, ais)
    found = table.groupby("case").label.max()
    print(f"{len(cases)} cases, true vessel among candidates in {found.mean() * 100:.1f}% of them")

    case_ids = np.array(sorted(table.case.unique()))
    rng = np.random.default_rng(1)
    rng.shuffle(case_ids)
    n_test = max(1, int(len(case_ids) * args.test_frac))
    test_ids, train_ids = set(case_ids[:n_test]), set(case_ids[n_test:])
    train, test = table[table.case.isin(train_ids)], table[table.case.isin(test_ids)].copy()

    model, model_name = make_model()
    model.fit(np.array([feature_vector(r) for r in train.to_dict("records")]), train.label.values)
    test["ml_score"] = model.predict_proba(np.array([feature_vector(r) for r in test.to_dict("records")]))[:, 1]

    n_test_cases = len(test_ids)
    metrics = {
        "model": model_name,
        "n_cases": len(cases), "n_test_cases": n_test_cases,
        "candidate_recall": round(float(found.mean()), 4),
        "rule_top1": topk_accuracy(test, "rule_score", n_test_cases, 1),
        "rule_top3": topk_accuracy(test, "rule_score", n_test_cases, 3),
        "ml_top1": topk_accuracy(test, "ml_score", n_test_cases, 1),
        "ml_top3": topk_accuracy(test, "ml_score", n_test_cases, 3),
    }
    if hasattr(model, "feature_importances_"):
        metrics["feature_importance"] = {k: round(float(v), 4) for k, v in zip(FEATURE_NAMES, model.feature_importances_)}

    try:
        test["lgbm_score"] = train_lgbm_ranker(train, test)
        metrics["lgbm_top1"] = topk_accuracy(test, "lgbm_score", n_test_cases, 1)
        metrics["lgbm_top3"] = topk_accuracy(test, "lgbm_score", n_test_cases, 3)
        metrics["lgbm_model"] = "LightGBM LambdaRank (learning-to-rank)"
    except Exception as exc:
        print(f"LightGBM ranker skipped: {exc}")

    try:
        from ml.attribution.attention_ranker import train as train_attention
        metrics.update(train_attention(table, train_ids, test_ids))
        metrics["attention_model"] = "Cross-attention (ships attend to spill + environment context)"
    except Exception as exc:
        print(f"Cross-attention model skipped: {exc}")

    RANKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RANKER_PATH, "wb") as fh:
        pickle.dump(model, fh)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "attribution.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
