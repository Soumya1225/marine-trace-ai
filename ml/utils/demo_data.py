"""Generate a small synthetic demo (SAR-like image + AIS CSV) off Huntington Beach.

Lets you run and test the whole system (pipeline, backend, frontend) before the real
datasets and the trained model are ready.

    python -m ml.utils.demo_data
Creates Datasets/demo/demo_sar.png, Datasets/demo/demo_ais.csv and Datasets/demo/demo_meta.json
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from ml.config import DATA_DIR, DEFAULT_ENV, WINDAGE
from ml.utils.geo import KM_PER_DEG_LAT, km_per_deg_lon

BBOX = (-118.25, 33.42, -117.85, 33.72)       # min_lon, min_lat, max_lon, max_lat
IMAGE_TIME = datetime(2021, 10, 3, 13, 52, tzinfo=timezone.utc)
W, H = 1400, 1050


def lonlat_to_px(lon, lat):
    x = (np.asarray(lon) - BBOX[0]) / (BBOX[2] - BBOX[0]) * W
    y = (BBOX[3] - np.asarray(lat)) / (BBOX[3] - BBOX[1]) * H
    return x, y


def straight_track(mmsi, name, vtype, length, start, end, t0, t1, rng, every_min=2, gap=None, slow=None):
    n = int((t1 - t0).total_seconds() / 60 / every_min) + 1
    f = np.linspace(0, 1, n)
    lat = start[0] + f * (end[0] - start[0]) + rng.normal(0, 0.0003, n)
    lon = start[1] + f * (end[1] - start[1]) + rng.normal(0, 0.0003, n)
    times = [t0 + timedelta(minutes=every_min * i) for i in range(n)]
    dist_km = np.hypot((end[0] - start[0]) * KM_PER_DEG_LAT, (end[1] - start[1]) * km_per_deg_lon(start[0]))
    speed = dist_km / ((t1 - t0).total_seconds() / 3600) / 1.852
    cog = (np.degrees(np.arctan2((end[1] - start[1]) * km_per_deg_lon(start[0]),
                                 (end[0] - start[0]) * KM_PER_DEG_LAT)) + 360) % 360
    sog = np.full(n, speed) + rng.normal(0, 0.3, n)
    if slow is not None:
        sog[slow[0]:slow[1]] = rng.uniform(1, 2.5, slow[1] - slow[0])
    df = pd.DataFrame({"MMSI": mmsi, "BaseDateTime": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in times],
                       "LAT": lat.round(5), "LON": lon.round(5), "SOG": sog.round(1), "COG": round(cog, 1),
                       "Heading": round(cog), "VesselName": name, "IMO": "", "CallSign": "", "VesselType": vtype,
                       "Status": 0, "Length": length, "Width": 30, "Draft": 10, "Cargo": vtype,
                       "TransceiverClass": "A"})
    if gap is not None:
        df = df.drop(df.index[gap[0]:gap[1]])
    return df


def main(out_dir=DATA_DIR / "demo", seed=7):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_end = IMAGE_TIME + timedelta(minutes=20)
    t_start = IMAGE_TIME - timedelta(hours=10)

    # the culprit tanker: sails NW->SE, slows down and switches AIS off while discharging
    culprit = straight_track(367000001, "OCEAN CRESCENT", 80, 228, (33.66, -118.20), (33.47, -117.88),
                             IMAGE_TIME - timedelta(hours=6), IMAGE_TIME - timedelta(hours=1), rng,
                             gap=(70, 85), slow=(55, 70))
    others = [
        straight_track(367000002, "PACIFIC TRADER", 70, 300, (33.45, -118.24), (33.70, -117.95), t_start, t_end, rng),
        straight_track(367000003, "SEA BREEZE", 60, 110, (33.70, -118.22), (33.70, -117.87), t_start + timedelta(hours=3), t_end, rng),
        straight_track(367000004, "HARBOR TUG 7", 52, 30, (33.60, -118.10), (33.62, -118.00), t_start + timedelta(hours=5), t_end, rng),
        straight_track(367000005, "GOLDEN NET", 30, 25, (33.50, -117.95), (33.44, -118.20), t_start, t_end, rng),
        straight_track(367000006, "NORDIC STAR", 80, 250, (33.43, -118.24), (33.43, -117.86), t_start, IMAGE_TIME - timedelta(hours=7), rng),
    ]
    ais = pd.concat([culprit] + others, ignore_index=True)
    ais.to_csv(out_dir / "demo_ais.csv", index=False)

    # SAR-like image: bright speckled sea with a dark elongated slick where the oil has drifted
    img = rng.gamma(4.0, 1.0, (H, W)).astype(np.float32)
    img = ndimage.gaussian_filter(img, 1.0) * 22 + 45
    slick = np.zeros((H, W), bool)
    u = DEFAULT_ENV["current_u"] + WINDAGE * DEFAULT_ENV["wind_u"]
    v = DEFAULT_ENV["current_v"] + WINDAGE * DEFAULT_ENV["wind_v"]
    times = pd.to_datetime(culprit.BaseDateTime).dt.tz_localize("UTC")
    release = culprit[(times >= IMAGE_TIME - timedelta(hours=4.5)) & (times <= IMAGE_TIME - timedelta(hours=2.8))]
    yy, xx = np.mgrid[0:H, 0:W]
    for _, p in release.iterrows():
        age = (IMAGE_TIME - pd.Timestamp(p.BaseDateTime, tz="UTC")).total_seconds()
        lat = p.LAT + v * age / 1000 / KM_PER_DEG_LAT
        lon = p.LON + u * age / 1000 / km_per_deg_lon(p.LAT)
        x, y = lonlat_to_px(lon, lat)
        r = 6 + age / 3600 * 2
        slick |= (xx - x) ** 2 + (yy - y) ** 2 < r * r
    slick = ndimage.binary_dilation(slick, iterations=2)
    img[slick] *= 0.35
    # a round low-wind look-alike patch
    img[(xx - 250) ** 2 + (yy - 850) ** 2 < 55 ** 2] *= 0.55
    # ships are bright radar targets: draw the AIS ships present at image time, plus one "dark" ship
    t_img = pd.Timestamp(IMAGE_TIME)
    ais_t = pd.to_datetime(ais.BaseDateTime).dt.tz_localize("UTC")
    ship_points = []
    for mmsi, g in ais.assign(t=ais_t).groupby("MMSI"):
        g = g[(g.t - t_img).abs() <= pd.Timedelta(minutes=5)]
        if len(g):
            ship_points.append((g.iloc[0].LON, g.iloc[0].LAT))
    dark_ship = (-118.06, 33.50)   # no AIS: a dark vessel
    ship_points.append(dark_ship)
    for lon, lat in ship_points:
        x, y = (int(v) for v in lonlat_to_px(lon, lat))
        if 4 <= x < W - 4 and 4 <= y < H - 4:
            img[y - 2:y + 3, x - 2:x + 3] = 255
            img[y, x - 12:x + 13] = 230
            img[y - 12:y + 13, x] = 230
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(out_dir / "demo_sar.png")

    meta = {"bbox": BBOX, "acquisition_time": IMAGE_TIME.isoformat(), "true_mmsi": 367000001,
            "dark_vessel": {"lon": dark_ship[0], "lat": dark_ship[1]},
            "note": "Synthetic demo data. Not real observations."}
    (out_dir / "demo_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Demo data written to {out_dir}")
    return meta


if __name__ == "__main__":
    main()
