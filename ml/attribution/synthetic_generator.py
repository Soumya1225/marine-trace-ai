"""Create synthetic spill cases from REAL AIS tracks, where the responsible vessel is known.

A random moving vessel "discharges" oil for 0.5-2 h; the oil drifts with a random wind/current
until a simulated satellite pass 1-8 h later. These cases are the ground truth for training
and evaluating the attribution engine (Top-1 / Top-3 accuracy).

    python -m ml.attribution.synthetic_generator Datasets/processed/ais_clean.pkl --cases 300
"""
import argparse
import pickle
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import SYNTHETIC_DIR, WINDAGE
from ml.features.ais_features import tracks
from ml.features.spill_features import describe_points
from ml.utils.geo import KM_PER_DEG_LAT, km_per_deg_lon


def random_env(rng):
    speed, direction = rng.uniform(2, 10), rng.uniform(0, 2 * np.pi)
    cur, cdir = rng.uniform(0, 0.3), rng.uniform(0, 2 * np.pi)
    return {"wind_u": speed * np.sin(direction), "wind_v": speed * np.cos(direction),
            "current_u": cur * np.sin(cdir), "current_v": cur * np.cos(cdir)}


def make_case(track, env, rng):
    track = track.sort_values("time").reset_index(drop=True)
    moving = track[track.sog.fillna(0) > 5]
    if len(moving) < 10:
        return None
    start_i = rng.integers(0, len(moving) - 5)
    release_h = rng.uniform(0.5, 2.0)
    t0 = moving.time.iloc[start_i]
    seg = track[(track.time >= t0) & (track.time <= t0 + pd.Timedelta(hours=release_h))]
    if len(seg) < 3:
        return None
    image_time = seg.time.iloc[-1] + timedelta(hours=float(rng.uniform(1, 8)))

    # each released point drifts from its release time until the image time
    u = env["current_u"] + WINDAGE * env["wind_u"]
    v = env["current_v"] + WINDAGE * env["wind_v"]
    lat, lon = [], []
    for _, p in seg.iterrows():
        dt = (image_time - p.time).total_seconds()
        n = 30
        spread = np.sqrt(2 * 10.0 * dt) / 1000.0   # km diffusion
        la = p.lat + (v * dt / 1000.0 + rng.normal(0, spread * 0.3, n)) / KM_PER_DEG_LAT
        lo = p.lon + (u * dt / 1000.0 + rng.normal(0, spread * 0.3, n)) / km_per_deg_lon(p.lat)
        lat.extend(la)
        lon.extend(lo)
    shape = describe_points(np.array(lat), np.array(lon))
    shape["area_km2"] = round(shape["length_km"] * shape["width_km"] * 0.6, 3)
    return {"spill": shape, "image_time": image_time, "env": env,
            "true_mmsi": int(track.mmsi.iloc[0]), "release_start": t0}


def generate(ais, n_cases=300, seed=42):
    rng = np.random.default_rng(seed)
    trs = [t for t in tracks(ais).values() if len(t) >= 20 and t.sog.median() > 5]
    if not trs:
        raise ValueError("No suitable moving vessel tracks found in the AIS data")
    cases, tries = [], 0
    while len(cases) < n_cases and tries < n_cases * 10:
        tries += 1
        case = make_case(trs[rng.integers(len(trs))], random_env(rng), rng)
        if case:
            cases.append(case)
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ais", help="cleaned AIS pickle from ml.preprocessing.ais_clean")
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--out", default=str(SYNTHETIC_DIR / "cases.pkl"))
    args = ap.parse_args()
    ais = pd.read_pickle(args.ais)
    cases = generate(ais, args.cases)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump({"cases": cases, "ais_path": args.ais}, fh)
    print(f"Saved {len(cases)} synthetic cases -> {args.out}")


if __name__ == "__main__":
    main()
