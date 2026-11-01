"""Score and rank candidate vessels for one spill (multimodal fusion of image + AIS features)."""
import pickle

import numpy as np
import pandas as pd

from ml.config import OVERLAP_KM, RANKER_PATH, RULE_WEIGHTS
from ml.features.ais_features import behaviour_features, track_to_json, vessel_summary
from ml.utils.geo import axis_angle_diff, point_to_polyline_km, to_local_km

FEATURE_NAMES = ["min_dist_km", "alignment", "overlap", "hours_before", "max_gap_h",
                 "speed_ratio", "loiter_h", "type_risk", "length_m"]


def vessel_features(track, spill, offset_fn):
    """Geometric + behaviour features of one vessel track relative to one spill."""
    axis = np.asarray(spill["axis"], float)            # [[lon, lat], ...]
    lat0, lon0 = spill["centroid_lat"], spill["centroid_lon"]
    hb = track.hours_before.clip(lower=0).values
    dlat, dlon = offset_fn(hb)
    # compare each AIS point with where the spill axis was at that moment (back-tracked)
    px, py = to_local_km(track.lat.values - dlat, track.lon.values - dlon, lat0, lon0)
    ax, ay = to_local_km(axis[:, 1], axis[:, 0], lat0, lon0)
    dists = point_to_polyline_km(px, py, ax, ay)
    i = int(np.argmin(dists))
    closest = track.iloc[i]

    cog = closest.cog if pd.notna(closest.cog) else closest.heading
    if pd.isna(cog) and len(track) > 1:  # derive course from neighbouring points
        j = min(i + 1, len(track) - 1)
        k = max(i - 1, 0)
        vx, vy = px[j] - px[k], py[j] - py[k]
        cog = np.degrees(np.arctan2(vx, vy)) % 360
    alignment = float(np.cos(np.radians(axis_angle_diff(cog, spill["orientation_deg"])))) if pd.notna(cog) else 0.5

    # how much of the spill axis lies on the vessel's (shifted) path
    near_pass = np.abs(hb - hb[i]) <= 2.0
    if near_pass.sum() >= 2:
        overlap = float(np.mean(point_to_polyline_km(ax, ay, px[near_pass], py[near_pass]) <= OVERLAP_KM))
    else:
        overlap = 0.0

    info = vessel_summary(track)
    beh = behaviour_features(track, focus_time=closest.time)
    return {
        **info, **beh,
        "min_dist_km": round(float(dists[i]), 3),
        "hours_before": round(float(hb[i]), 2),
        "closest_time": closest.time.isoformat(),
        "alignment": round(alignment, 3),
        "overlap": round(overlap, 3),
        "length_m": info["length_m"] or 0.0,
    }


def rule_score(f):
    proximity = np.exp(-f["min_dist_km"] / 2.0)
    recency = np.exp(-f["hours_before"] / 6.0)
    behaviour = max(1.0 if f["ais_gap_flag"] else 0.0, 1.0 if f["slowdown_flag"] else 0.0,
                    min(f["loiter_h"] / 2.0, 1.0))
    parts = {
        "proximity": proximity, "alignment": f["alignment"] * proximity,
        "overlap": f["overlap"], "recency": recency * proximity,
        "type_risk": f["type_risk"], "behaviour": behaviour * proximity,
    }
    return float(sum(RULE_WEIGHTS[k] * v for k, v in parts.items())), parts


def evidence_text(f):
    lines = [f"Passed {f['min_dist_km']:.1f} km from the spill's back-tracked position "
             f"{f['hours_before']:.1f} h before the satellite image."]
    if f["alignment"] > 0.85:
        lines.append("Its course matches the spill's long axis (typical of a discharge while underway).")
    if f["overlap"] > 0.3:
        lines.append(f"{f['overlap'] * 100:.0f}% of the spill axis lies on this vessel's path.")
    if f["ais_gap_flag"]:
        lines.append(f"AIS transmission gap of {f['max_gap_h']:.1f} h near the spill.")
    if f["slowdown_flag"]:
        lines.append("Slowed down sharply near the spill.")
    if f["loiter_h"] > 0.5:
        lines.append(f"Loitered for {f['loiter_h']:.1f} h at low speed.")
    if f["type_risk"] >= 1.0:
        lines.append("Vessel type: tanker (higher discharge risk).")
    return lines


def load_ranker(path=RANKER_PATH):
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def feature_vector(f):
    return [float(f.get(k) or 0.0) for k in FEATURE_NAMES]


def shap_contributions(ranker, X):
    """Per-ship SHAP values (log-odds) of the XGBoost ranker: how much each feature pushed the score up or down.
    Uses XGBoost's built-in TreeSHAP (pred_contribs), so no extra package is needed."""
    try:
        import xgboost as xgb
        booster = ranker.get_booster() if hasattr(ranker, "get_booster") else ranker
        names = getattr(booster, "feature_names", None)   # set if the model was trained on a DataFrame
        contrib = booster.predict(xgb.DMatrix(np.asarray(X, float), feature_names=names), pred_contribs=True)
        return contrib[:, :-1], contrib[:, -1]          # last column is the base value (average)
    except Exception as exc:
        print(f"[SHAP] skipped: {exc}")
        return None, None


def rank_vessels(candidates, spill, offset_fn, ranker=None, top_k=10, include_tracks=True, context=None):
    """candidates: AIS rows from candidate_vessels(). Returns ranked suspect dicts."""
    if candidates is None or candidates.empty:
        return []
    results = []
    for mmsi, track in candidates.groupby("mmsi"):
        track = track.sort_values("time").reset_index(drop=True)
        f = vessel_features(track, spill, offset_fn)
        f["rule_score"], f["score_parts"] = rule_score(f)
        f["evidence"] = evidence_text(f)
        if include_tracks:
            f["track"] = track_to_json(track)
        results.append(f)

    if ranker is not None:
        X = np.array([feature_vector(f) for f in results])
        probs = ranker.predict_proba(X)[:, 1]
        for f, p in zip(results, probs):
            f["ml_score"] = round(float(p), 4)
        shap_vals, base = shap_contributions(ranker, X)
        if shap_vals is not None:
            for f, row, b0 in zip(results, shap_vals, base):
                f["shap"] = {k: round(float(v), 4) for k, v in zip(FEATURE_NAMES, row)}
                f["shap_base"] = round(float(b0), 4)
    if context is not None:
        from ml.attribution.attention_ranker import score as attention_score
        probs = attention_score(results, context)
        if probs is not None:
            for f, p in zip(results, probs):
                f["attention_score"] = round(float(p), 4)
    for f in results:
        # blend: the ML model was trained on synthetic cases, the rule score encodes physical evidence
        f["score"] = 0.5 * f["rule_score"] + 0.5 * f["ml_score"] if "ml_score" in f else f["rule_score"]

    results.sort(key=lambda f: f["score"], reverse=True)
    total = sum(f["score"] for f in results) or 1.0
    for rank, f in enumerate(results, start=1):
        f["rank"] = rank
        f["confidence"] = round(f["score"] / total, 4)   # share of suspicion among candidates
        f["level"] = "high" if f["score"] >= 0.6 else "medium" if f["score"] >= 0.35 else "low"
        f["rule_score"] = round(f["rule_score"], 4)
        f["score"] = round(f["score"], 4)
        f["score_parts"] = {k: round(float(v), 3) for k, v in f["score_parts"].items()}
    return results[:top_k]
