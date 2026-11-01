"""End-to-end analysis: detection -> tracking -> source identification -> prediction -> impact -> decision.

    python -m ml.pipeline --scene Datasets/raw/sentinel1/<scene>.zip \
        --ais Datasets/raw/ais/AIS_2021_10_02.csv Datasets/raw/ais/AIS_2021_10_03.csv \
        --bbox -118.6 33.3 -117.6 33.9 --out Datasets/processed/run1
"""
import argparse
import json
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ml.attribution.attribute import attribute_spill
from ml.attribution.drift import ENSEMBLE, Environment, PerturbedEnvironment, seed_particles, simulate
from ml.attribution.scorer import load_ranker
from ml.config import (ATTRIBUTION_WINDOW_H, CLASSES, DISAGREEMENT_FLAG, FORECAST_HOURS, LOOKALIKE_CLASS,
                       MAX_RELIABLE_WIND_MS, MAX_SPILL_KM2, MIN_RELIABLE_WIND_MS, OIL_CLASS)
from ml.decision.support import recommend
from ml.features.infrastructure import action_for, assess_spill, mark_known_platforms
from ml.features.ship_detection import dark_vessels_near, detect_ships, match_dark_vessels
from ml.features.spill_features import extract_spills
from ml.impact.assessment import assess, load_coastline, load_sites
from ml.preprocessing.ais_clean import clean_ais, load_ais
from ml.preprocessing.land_mask import land_mask
from ml.preprocessing.sar_preprocess import load_scene
from ml.segmentation.predict import SpillDetector
from ml.utils.geo import KM_PER_DEG_LAT, apply_affine, convex_hull_lonlat, km_per_deg_lon


def wind_speed(env, lat, lon, when):
    wu, wv, _, _ = env.at(np.array([lat]), np.array([lon]), when)
    return float(np.hypot(wu, wv))


def reject_calm_areas(class_map, transform, scene_wind_ms, wind_at=None):
    """Very large dark areas in calm wind (low-wind areas) or in very strong wind (rain cells, wind shadows,
    monsoon squalls) are look-alikes, not oil: relabel them. In normal wind a big slick is kept.
    wind_at(lat, lon) gives the wind at each dark area itself (wind can differ a lot across one image);
    without it the scene wind is used for every area."""
    from scipy import ndimage
    a, b, _, d, e, _ = transform[:6]
    labels, n = ndimage.label(class_map == OIL_CLASS)
    if n == 0:
        return class_map, 0
    sizes = ndimage.sum(np.ones_like(labels), labels, range(1, n + 1))
    lat_mid = transform[5] + e * class_map.shape[0] / 2
    px_km2 = abs(a * e - b * d) * km_per_deg_lon(lat_mid) * KM_PER_DEG_LAT
    big = [i for i, size in enumerate(sizes, start=1) if size * px_km2 > MAX_SPILL_KM2]
    if not big:
        return class_map, 0
    centres = ndimage.center_of_mass(np.ones_like(labels), labels, big)
    rejected = 0
    for i, (r, col) in zip(big, centres):
        ws = scene_wind_ms
        if wind_at is not None:
            lon, lat = apply_affine(transform, col + 0.5, r + 0.5)
            try:
                ws = wind_at(lat, lon)
            except Exception:                           # noqa: BLE001 - fall back to the scene wind
                ws = scene_wind_ms
        if not (MIN_RELIABLE_WIND_MS <= ws <= MAX_RELIABLE_WIND_MS):
            class_map[labels == i] = LOOKALIKE_CLASS
            rejected += 1
    return class_map, rejected


def save_overlays(scene, class_map, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(scene["image_u8"]).save(out_dir / "sar.png")
    rgba = np.zeros(class_map.shape + (4,), np.uint8)
    rgba[class_map == OIL_CLASS] = [230, 40, 30, 190]
    rgba[class_map == LOOKALIKE_CLASS] = [250, 200, 0, 110]
    Image.fromarray(rgba, "RGBA").save(out_dir / "mask.png")
    return {"sar": "sar.png", "mask": "mask.png"}


def forecast_spill(spill, image_time, env, coast, hours=max(FORECAST_HOURS)):
    seeds = (spill.get("polygon") or []) + spill["axis"]
    lat, lon = seed_particles(seeds, n=400)
    snaps = simulate(lat, lon, image_time, hours, env, direction=+1, coast=coast)
    # ensemble: same spill, wind a bit stronger/weaker and turned left/right -> uncertainty zone
    members = []
    for k, (factor, angle) in enumerate(ENSEMBLE):
        la, lo = seed_particles(seeds, n=120, rng=np.random.default_rng(k + 1))
        members.append(simulate(la, lo, image_time, hours, PerturbedEnvironment(env, factor, angle),
                                direction=+1, coast=coast, seed=k + 1))
    from ml.attribution.lstm_drift import forecast as lstm_forecast
    lstm = lstm_forecast(spill["centroid_lat"], spill["centroid_lon"], image_time, env, hours=24)
    lstm_at = {h: (la, lo) for h, la, lo in lstm} if lstm else {}
    summary = []
    for h in FORECAST_HOURS:
        s = min(snaps, key=lambda x: abs(x["hours"] - h))
        ms = [min(m, key=lambda x: abs(x["hours"] - h)) for m in members]
        all_lat = np.concatenate([s["lat"]] + [m["lat"] for m in ms])
        all_lon = np.concatenate([s["lon"]] + [m["lon"] for m in ms])
        summary.append({
            "hours": h,
            "time": s["time"].isoformat(),
            "centroid_lat": round(float(np.mean(s["lat"])), 6),
            "centroid_lon": round(float(np.mean(s["lon"])), 6),
            "beached_fraction": round(float(np.mean(s["beached"])), 3),
            "polygon": convex_hull_lonlat(s["lat"], s["lon"]),
            "uncertainty_polygon": convex_hull_lonlat(all_lat, all_lon),
            "lstm_centroid": list(lstm_at[h]) if h in lstm_at else None,
            "particles": [[round(float(lo), 5), round(float(la), 5)] for lo, la in zip(s["lon"][::4], s["lat"][::4])],
        })
    return summary, snaps


def analyze(scene_path, ais=None, bbox=None, acquisition_time=None, env_cfg=None, out_dir="Datasets/processed/run",
            detector=None, ranker=None, progress=lambda pct, msg: None):
    out_dir = Path(out_dir)
    progress(5, "Loading satellite image")
    if Path(scene_path).suffix.lower() in (".png", ".jpg", ".jpeg"):
        # plain pictures can be anything: make sure it is a radar image of the sea before running the models
        from ml.quickcheck import gate, load_image
        g = gate(load_image(Path(scene_path).read_bytes()))
        if not g["passed"]:
            raise ValueError("Not a satellite radar image, so it cannot be analysed: " + "; ".join(g["reasons"]))
    scene = load_scene(scene_path, bbox=bbox, acquisition_time=acquisition_time)
    image_time = scene["acquisition_time"]
    if image_time is None:
        raise ValueError("Acquisition time unknown: provide it for this image")
    image_time = image_time.astimezone(timezone.utc)

    progress(10, "Getting wind and current data for this area and date")
    env = Environment.auto(bbox=scene["bounds"], when=image_time, **(env_cfg or {}))
    min_lon, min_lat, max_lon, max_lat = scene["bounds"]
    scene_wind = wind_speed(env, (min_lat + max_lat) / 2, (min_lon + max_lon) / 2, image_time)

    progress(20, "Detecting oil spills")
    detector = detector or SpillDetector()
    class_map, oil_prob = detector.predict(scene["image_u8"])
    land = land_mask(class_map.shape, scene["transform"])
    class_map[land] = 0                                   # no oil on land
    from scipy import ndimage
    valid = ndimage.binary_erosion(scene["image_u8"] > 0, iterations=6)
    class_map[~valid] = 0                                 # no oil along the image's no-data edge
    class_map, rejected = reject_calm_areas(class_map, scene["transform"], scene_wind,
                                            wind_at=lambda la, lo: wind_speed(env, la, lo, image_time))
    uncertainty = getattr(detector, "last_uncertainty", None)
    spills = extract_spills(class_map, scene["transform"], image_time, oil_prob, uncertainty=uncertainty)
    for sp in spills:
        ws = wind_speed(env, sp["centroid_lat"], sp["centroid_lon"], image_time)
        notes = []
        if ws < MIN_RELIABLE_WIND_MS:
            notes.append(f"Low wind ({ws:.1f} m/s): calm water also looks dark in radar, so this may be a look-alike.")
        elif ws > MAX_RELIABLE_WIND_MS:
            notes.append(f"Very strong wind ({ws:.1f} m/s): real slicks break up and mix into the water, and rain cells "
                         "or wind shadows look dark in radar, so this may be a look-alike.")
        if sp.get("confidence") is not None and sp["confidence"] < 0.5:
            notes.append("Model confidence is below 0.5.")
        if sp.get("model_disagreement") is not None and sp["model_disagreement"] > DISAGREEMENT_FLAG:
            notes.append(f"The ensemble models disagree here (spread {sp['model_disagreement']:.2f}).")
        sp["wind_speed_ms"] = round(ws, 1)
        sp["reliability"] = "low" if notes else "normal"
        sp["reliability_notes"] = notes
    # second stage: physical evidence (wind, contrast, edges, shape, texture) -> oil vs look-alike score
    from ml.lookalike.second_stage import check_spills
    check_spills(scene["image_u8"], class_map == OIL_CLASS, scene["transform"], spills)
    overlays = save_overlays(scene, class_map, out_dir)
    if uncertainty is not None:            # purple where the models disagree
        u = np.clip(uncertainty / 0.4, 0, 1)
        rgba = np.zeros(u.shape + (4,), np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = 120, 40, 170
        rgba[..., 3] = (u * 200).astype(np.uint8)
        Image.fromarray(rgba, "RGBA").save(out_dir / "uncertainty.png")
        overlays["uncertainty"] = "uncertainty.png"

    progress(40, "Loading AIS vessel data")
    pad = 0.4
    area = (min_lon - pad, min_lat - pad, max_lon + pad, max_lat + pad)
    if isinstance(ais, pd.DataFrame):
        ais_df = ais.copy()
        if not pd.api.types.is_datetime64_any_dtype(ais_df["time"]):
            ais_df["time"] = pd.to_datetime(ais_df["time"], utc=True)
            ais_df = clean_ais(ais_df)
    elif ais:
        ais_df = load_ais(ais, bbox=area, t_start=image_time - pd.Timedelta(hours=ATTRIBUTION_WINDOW_H + 1),
                          t_end=image_time + pd.Timedelta(hours=1))
    else:
        ais_df = pd.DataFrame(columns=["mmsi", "time", "lat", "lon", "sog", "cog", "heading",
                                       "vessel_name", "imo", "vessel_type", "length"])

    progress(42, "Looking for ships without AIS (dark vessels)")
    targets = detect_ships(scene["db"], land, scene["transform"])
    ships = match_dark_vessels(targets, ais_df, image_time) if not ais_df.empty else [dict(t, dark=True) for t in targets]
    # a target at the same spot in another image of this area (hours apart) is a platform, not a dark ship
    from ml.features.fixed_structures import mark_fixed, other_scene_targets, ship_counts
    mark_fixed(ships, other_scene_targets(out_dir.parent, image_time, scene["bounds"], exclude_dir=out_dir))
    mark_known_platforms(ships)                           # targets on a known oil platform are the platform
    coast = load_coastline(bbox=(area[0] - 1, area[1] - 1, area[2] + 1, area[3] + 1))
    sites = load_sites()
    ranker = ranker if ranker is not None else load_ranker()

    results = []
    for i, spill in enumerate(spills[:10]):
        progress(45 + int(45 * i / max(len(spills[:10]), 1)), f"Analysing spill {i + 1} of {len(spills[:10])}")
        spill["nearby_dark_vessels"] = dark_vessels_near(ships, spill["centroid_lat"], spill["centroid_lon"])
        wu, wv, cu, cv = env.at(np.array([spill["centroid_lat"]]), np.array([spill["centroid_lon"]]), image_time)
        context = {"ctx_area_km2": spill["area_km2"], "ctx_elongation": spill["elongation"],
                   "ctx_length_km": spill["length_km"], "ctx_wind_ms": float(np.hypot(wu, wv)),
                   "ctx_current_ms": float(np.hypot(cu, cv))}
        suspects, back_path = attribute_spill(spill, ais_df, image_time, env, ranker=ranker, context=context) \
            if not ais_df.empty else ([], [])
        forecast, snaps = forecast_spill(spill, image_time, env, coast)
        impact = assess(spill, snaps, sites)
        spill["infrastructure"] = assess_spill(spill, back_path)       # platforms / pipelines nearby
        decision = recommend(spill, impact, suspects, forecast)
        extra = action_for(spill["infrastructure"])
        if extra:
            decision["actions"].insert(1, extra)
        results.append({"spill": spill, "suspects": suspects, "backtrack": back_path,
                        "forecast": forecast, "impact": impact, "decision": decision})

    progress(95, "Saving results")
    lookalike_px = int((class_map == LOOKALIKE_CLASS).sum())
    result = {
        "scene": {"source": scene["source"], "calibration": scene.get("calibration"),
                  "sigma0_median_db": (round(float(np.nanmedian(scene["sigma0_db"])), 2)
                                       if scene.get("sigma0_db") is not None else None),
                  "acquisition_time": image_time.isoformat(),
                  "bounds": scene["bounds"], "overlays": overlays, "shape": list(class_map.shape)},
        "detector": {"mode": detector.mode, "model": detector.arch, "classes": CLASSES},
        "ais": {"points": int(len(ais_df)), "vessels": int(ais_df.mmsi.nunique()) if len(ais_df) else 0},
        "environment": env.describe((min_lat + max_lat) / 2, (min_lon + max_lon) / 2, image_time),
        "lookalike_pixels": lookalike_px,
        "scene_wind_ms": round(scene_wind, 1),
        "land_masked_pixels": int(land.sum()),
        "calm_areas_rejected": rejected,
        "ships": {**ship_counts(ships), "targets": ships},
        "spills": results,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    progress(100, "Done")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ais", nargs="*", default=[])
    ap.add_argument("--bbox", nargs=4, type=float)
    ap.add_argument("--time", help="acquisition time (ISO) if not in the file name")
    ap.add_argument("--out", default="Datasets/processed/run")
    args = ap.parse_args()
    res = analyze(args.scene, args.ais, tuple(args.bbox) if args.bbox else None, args.time, out_dir=args.out,
                  progress=lambda p, m: print(f"[{p:3d}%] {m}"))
    print(f"Scene wind {res['scene_wind_ms']} m/s | land pixels masked {res['land_masked_pixels']} | "
          f"large calm areas rejected {res['calm_areas_rejected']} | spills kept {len(res['spills'])}")
    sh = res["ships"]
    print(f"Ships seen by radar: {sh['detected']} | matched to AIS: {sh['matched']} | dark (no AIS): {sh['dark']} | "
          f"fixed structures: {sh.get('fixed', 0)}")
    for r in res["spills"]:
        s = r["spill"]
        print(f"Spill {s['spill_no']}: {s['area_km2']} km2 at {s['centroid_lat']:.4f},{s['centroid_lon']:.4f} "
              f"-> priority {r['decision']['priority']} (wind {s['wind_speed_ms']} m/s, reliability {s['reliability']})")
        la = s.get("lookalike_check")
        if la:
            print(f"   look-alike check: {la['verdict']} (oil score {la['oil_score']})")
        inf = s.get("infrastructure") or {}
        if inf.get("possible_non_ship_source"):
            print("   possible non-ship source: " + inf["note"])
        for v in r["suspects"][:3]:
            print(f"   #{v['rank']} {v['vessel_name']} (MMSI {v['mmsi']}) score {v['score']}")
        if s.get("excluded_responders"):
            print("   excluded responders: " + ", ".join(x["vessel_name"] for x in s["excluded_responders"][:6]))


if __name__ == "__main__":
    main()
