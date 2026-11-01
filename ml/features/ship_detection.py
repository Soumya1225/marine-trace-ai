"""Dark-vessel detection.

Ships are strong radar reflectors: they appear as small, very bright spots in a SAR image.
We detect those bright targets at sea (CFAR-style: much brighter than the local background) and
compare them with where AIS says ships were at the image time. A bright target with no AIS ship
nearby is a "dark vessel" - a ship that is not transmitting AIS, which is typical of illegal discharge.
"""
import numpy as np
import pandas as pd
from scipy import ndimage

from ml.utils.geo import KM_PER_DEG_LAT, apply_affine, haversine_km, km_per_deg_lon


METHOD = "global"      # "local": compare with the surrounding sea (CFAR); "global": compare with the whole image
LOCAL_MIN_DB = 3.0    # local method: candidate pixels are at least this many dB above the local sea
LOCAL_CORE_DB = 6.0   # local method: a ship / platform has a core at least this many dB above the local sea


def detect_ships(db, land, transform, k=5.0, z_min=6.0, window=61, min_px=2, max_area_km2=1.0, core_z=10.0,
                 method=None, local_min_db=None, local_core_db=None):
    """Return bright point targets at sea: [{lat, lon, pixels, brightness}].

    Uses the unclipped backscatter image (dB), so very bright ships are not flattened by the display stretch.
    method="local" (default) judges each spot against its own surroundings, which keeps working when the sea is
    rough or its brightness changes across the image; method="global" uses the whole image's median (older)."""
    method = method or METHOD
    local_min_db = LOCAL_MIN_DB if local_min_db is None else local_min_db
    local_core_db = LOCAL_CORE_DB if local_core_db is None else local_core_db
    finite = np.isfinite(db)
    img = np.where(finite, db, np.nanmedian(db) if finite.any() else 0).astype(np.float32)
    valid = ndimage.binary_erosion(finite, iterations=10) & ~ndimage.binary_dilation(land, iterations=8)
    mean = ndimage.uniform_filter(img, window)
    sq = ndimage.uniform_filter(img * img, window)
    std = np.sqrt(np.maximum(sq - mean * mean, 1e-6))
    if not valid.any():
        return []
    # robust brightness score against the whole sea (median / MAD), so speckle peaks are ignored
    sea = img[valid]
    med = np.median(sea)
    mad = 1.4826 * np.median(np.abs(sea - med)) + 1e-6
    contrast = img - mean                                  # dB above the local sea (61 x 61 window)
    if method == "local":
        bright = (img > mean + k * std) & (contrast >= local_min_db) & valid
    else:
        bright = (img > mean + k * std) & ((img - med) / mad > z_min) & valid
    # a ship's bright cross often splits into pieces: join pieces closer than ~3 pixels
    labels, n = ndimage.label(ndimage.binary_dilation(bright, iterations=2))
    labels = labels * bright
    zmap = (img - med) / mad
    if n == 0:
        return []
    a, b, _, d, e, _ = transform
    lat_mid = transform[5] + e * img.shape[0] / 2
    px_km2 = abs(a * e - b * d) * km_per_deg_lon(lat_mid) * KM_PER_DEG_LAT
    targets = []
    for idx, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        region = labels[sl] == idx
        size = int(region.sum())
        if size < min_px or size * px_km2 > max_area_km2:
            continue
        # a real ship has a very bright core (strong reflector), not just a slightly bright patch
        if method == "local":
            if contrast[sl][region].max() < local_core_db:
                continue
        elif zmap[sl][region].max() < core_z:
            continue
        rows, cols = np.nonzero(region)
        vals = img[sl][region]
        r = np.average(rows + sl[0].start, weights=vals) + 0.5
        c = np.average(cols + sl[1].start, weights=vals) + 0.5
        lon, lat = apply_affine(transform, c, r)
        targets.append({"lat": round(float(lat), 6), "lon": round(float(lon), 6), "pixels": size,
                        "brightness": round(float(vals.max()), 1)})
    return targets


def ais_positions_at(ais, when, window_min=30):
    """Best estimate of every vessel's position at `when` (interpolated or nearest report)."""
    if ais is None or ais.empty:
        return pd.DataFrame(columns=["mmsi", "vessel_name", "lat", "lon", "sog", "dt_h"])
    when = pd.Timestamp(when)
    w = pd.Timedelta(minutes=window_min)
    near = ais[(ais.time >= when - w) & (ais.time <= when + w)]
    rows = []
    for mmsi, g in near.groupby("mmsi"):
        g = g.sort_values("time")
        before, after = g[g.time <= when], g[g.time >= when]
        if len(before) and len(after):
            p, q = before.iloc[-1], after.iloc[0]
            span = (q.time - p.time).total_seconds() or 1.0
            f = (when - p.time).total_seconds() / span
            lat, lon, dt_h = p.lat + f * (q.lat - p.lat), p.lon + f * (q.lon - p.lon), 0.0
            src = p
        else:
            src = g.iloc[(g.time - when).abs().argmin()]
            lat, lon, dt_h = src.lat, src.lon, abs((src.time - when).total_seconds()) / 3600
        rows.append({"mmsi": int(mmsi), "vessel_name": str(src.get("vessel_name") or ""), "lat": float(lat),
                     "lon": float(lon), "sog": float(src.sog) if pd.notna(src.sog) else 0.0, "dt_h": dt_h})
    return pd.DataFrame(rows)


def match_dark_vessels(targets, ais, when, base_km=0.6):
    """Label each radar target as matched to an AIS vessel or 'dark'."""
    pos = ais_positions_at(ais, when)
    out = []
    for t in targets:
        item = dict(t)
        if len(pos):
            d = haversine_km(t["lat"], t["lon"], pos.lat.values, pos.lon.values)
            # allow for AIS timing: a ship moves sog*1.852 km per hour between reports
            allowed = base_km + pos.sog.values * 1.852 * pos.dt_h.values
            i = int(np.argmin(d - allowed))
            if d[i] <= allowed[i]:
                item.update({"dark": False, "mmsi": int(pos.mmsi.iloc[i]), "vessel_name": pos.vessel_name.iloc[i],
                             "match_km": round(float(d[i]), 2)})
                out.append(item)
                continue
            item["nearest_ais_km"] = round(float(d.min()), 2)
        item["dark"] = True
        out.append(item)
    return out


def dark_vessels_near(ships, lat, lon, radius_km=15.0):
    near = []
    for s in ships:
        if s.get("dark"):
            dist = float(haversine_km(lat, lon, s["lat"], s["lon"]))
            if dist <= radius_km:
                near.append({**s, "distance_km": round(dist, 2)})
    return sorted(near, key=lambda x: x["distance_km"])
