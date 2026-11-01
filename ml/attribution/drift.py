"""Lagrangian particle drift model used for back-tracking (attribution) and forecasting (prediction).

Oil velocity = surface current + WINDAGE * wind + random-walk diffusion.
Wind/current come from NetCDF files if available (ERA5 u10/v10, Copernicus Marine uo/vo),
otherwise from constant values (config DEFAULT_ENV or values typed by the analyst).
"""
from datetime import timedelta
from pathlib import Path

import numpy as np

from ml.config import (BEACH_DISTANCE_KM, DEFAULT_ENV, DIFFUSION_M2S, DRIFT_STEP_MIN,
                       N_PARTICLES, RAW_DIR, WINDAGE)
from ml.utils.geo import KM_PER_DEG_LAT, km_per_deg_lon, to_local_km


class Environment:
    def __init__(self, wind_u=None, wind_v=None, current_u=None, current_v=None,
                 wind_file=None, current_file=None):
        self.wind_u = DEFAULT_ENV["wind_u"] if wind_u is None else float(wind_u)
        self.wind_v = DEFAULT_ENV["wind_v"] if wind_v is None else float(wind_v)
        self.current_u = DEFAULT_ENV["current_u"] if current_u is None else float(current_u)
        self.current_v = DEFAULT_ENV["current_v"] if current_v is None else float(current_v)
        self.wind_ds = self._open(wind_file)
        self.current_ds = self._open(current_file)
        self.wind_file = Path(wind_file).name if wind_file and self.wind_ds is not None else None
        self.current_file = Path(current_file).name if current_file and self.current_ds is not None else None
        self.notes = []

    @staticmethod
    def _open(path):
        if not path or not Path(path).exists():
            return None
        try:
            import xarray as xr
            return xr.open_dataset(path)
        except Exception as exc:
            print(f"[Environment] could not open {path}: {exc}")
            return None

    @classmethod
    def auto(cls, bbox=None, when=None, **overrides):
        """Wind / current files for this image.

        With the image area and time (bbox, when): only files that really cover them are used, and missing ones
        are downloaded automatically (ml/environment_auto.py). Without them: the first NetCDF files found in
        Datasets/raw/weather (old behaviour, used by scripts and tests)."""
        if bbox is not None and when is not None:
            from ml.environment_auto import prepare
            typed = {k: overrides.get(k) for k in ("wind_u", "wind_v", "current_u", "current_v")}
            res = prepare(bbox, when,
                          need_wind=typed["wind_u"] is None or typed["wind_v"] is None,
                          need_current=typed["current_u"] is None or typed["current_v"] is None)
            env = cls(wind_file=res["wind_file"], current_file=res["current_file"], **overrides)
            env.notes = res["notes"]
            return env
        weather = RAW_DIR / "weather"
        wind = next(iter(sorted(weather.glob("*wind*.nc")) + sorted(weather.glob("era5*.nc"))), None) if weather.exists() else None
        cur = next(iter(sorted(weather.glob("*current*.nc")) + sorted(weather.glob("cmems*.nc"))), None) if weather.exists() else None
        return cls(wind_file=wind, current_file=cur, **overrides)

    def describe(self, lat=None, lon=None, when=None):
        """Values actually used: sampled from the NetCDF files at (lat, lon, when) when given."""
        wu, wv, cu, cv = self.wind_u, self.wind_v, self.current_u, self.current_v
        if lat is not None and when is not None:
            wu, wv, cu, cv = self.at(np.array([lat]), np.array([lon]), when)
        return {
            "source": "netcdf" if (self.wind_ds is not None or self.current_ds is not None) else "constant",
            "wind_u": round(float(wu), 2), "wind_v": round(float(wv), 2),
            "current_u": round(float(cu), 3), "current_v": round(float(cv), 3),
            "wind_source": self.wind_file or "default / typed value",
            "current_source": self.current_file or "default / typed value",
            "notes": list(getattr(self, "notes", [])),
        }

    @staticmethod
    def _sample(ds, names, lat, lon, when):
        import pandas as pd
        u_name, v_name = names
        time_dim = "valid_time" if "valid_time" in ds.dims else "time"
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        sel = {time_dim: pd.Timestamp(when).tz_localize(None) if pd.Timestamp(when).tzinfo else pd.Timestamp(when),
               lat_name: float(np.mean(lat)), lon_name: float(np.mean(lon))}
        sub = ds.sel(sel, method="nearest")
        if "depth" in sub.dims:
            sub = sub.isel(depth=0)
        return float(sub[u_name].values), float(sub[v_name].values)

    def at(self, lat, lon, when):
        """Return (wind_u, wind_v, current_u, current_v) in m/s for a particle cloud."""
        wu, wv, cu, cv = self.wind_u, self.wind_v, self.current_u, self.current_v
        try:
            if self.wind_ds is not None:
                wu, wv = self._sample(self.wind_ds, ("u10", "v10"), lat, lon, when)
            if self.current_ds is not None:
                cu, cv = self._sample(self.current_ds, ("uo", "vo"), lat, lon, when)
        except Exception:
            pass
        wu, wv, cu, cv = (0.0 if not np.isfinite(v) else v for v in (wu, wv, cu, cv))
        return wu, wv, cu, cv


class PerturbedEnvironment:
    """Same environment with the wind rotated and scaled: used for ensemble (uncertainty) forecasts."""

    def __init__(self, base, speed_factor=1.0, angle_deg=0.0):
        self.base, self.k, self.a = base, speed_factor, np.radians(angle_deg)

    def at(self, lat, lon, when):
        wu, wv, cu, cv = self.base.at(lat, lon, when)
        ca, sa = np.cos(self.a), np.sin(self.a)
        return self.k * (wu * ca - wv * sa), self.k * (wu * sa + wv * ca), cu, cv


ENSEMBLE = [(0.8, -20.0), (1.2, -20.0), (0.8, 20.0), (1.2, 20.0)]   # wind speed x, direction +- degrees


def seed_particles(points_lonlat, n=N_PARTICLES, jitter_km=0.2, rng=None):
    """Seed particles on/around given [lon, lat] points (spill polygon or axis)."""
    rng = rng or np.random.default_rng(0)
    pts = np.asarray(points_lonlat, dtype=float)
    idx = rng.integers(0, len(pts), n)
    lon, lat = pts[idx, 0], pts[idx, 1]
    lat = lat + rng.normal(0, jitter_km, n) / KM_PER_DEG_LAT
    lon = lon + rng.normal(0, jitter_km, n) / km_per_deg_lon(np.mean(lat))
    return lat, lon


def simulate(lat, lon, start_time, hours, env, direction=+1, step_min=DRIFT_STEP_MIN,
             windage=WINDAGE, diffusion=DIFFUSION_M2S, coast=None, seed=0):
    """Move particles forward (direction=+1) or backward (-1) in time.

    coast: optional (lat_array, lon_array) coastline points; particles reaching it stop (beached).
    Returns a list of snapshots: {"hours": h, "time": t, "lat": arr, "lon": arr, "beached": bool arr}.
    """
    rng = np.random.default_rng(seed)
    lat, lon = np.array(lat, float), np.array(lon, float)
    beached = np.zeros(len(lat), bool)
    dt = step_min * 60.0
    n_steps = int(round(hours * 60 / step_min))
    tree, lat0, lon0 = None, float(np.mean(lat)), float(np.mean(lon))
    if coast is not None and len(coast[0]) and direction > 0:
        from scipy.spatial import cKDTree
        cx, cy = to_local_km(coast[0], coast[1], lat0, lon0)
        tree = cKDTree(np.column_stack([cx, cy]))
    snaps = [{"hours": 0.0, "time": start_time, "lat": lat.copy(), "lon": lon.copy(), "beached": beached.copy()}]
    t = start_time
    for step in range(1, n_steps + 1):
        wu, wv, cu, cv = env.at(lat, lon, t)
        u = cu + windage * wu
        v = cv + windage * wv
        sigma = np.sqrt(2 * diffusion * dt)
        dx = direction * u * dt + rng.normal(0, sigma, len(lat))
        dy = direction * v * dt + rng.normal(0, sigma, len(lat))
        move = ~beached
        lat[move] += dy[move] / 1000.0 / KM_PER_DEG_LAT
        lon[move] += dx[move] / 1000.0 / km_per_deg_lon(lat[move])
        t = t + direction * timedelta(seconds=dt)
        if tree is not None:
            px, py = to_local_km(lat, lon, lat0, lon0)
            dist, _ = tree.query(np.column_stack([px, py]))
            beached |= dist < BEACH_DISTANCE_KM
        snaps.append({"hours": step * step_min / 60.0, "time": t, "lat": lat.copy(), "lon": lon.copy(),
                      "beached": beached.copy()})
    return snaps


def centroid_track(snaps):
    """Mean particle position per snapshot -> arrays (hours, lat, lon)."""
    h = np.array([s["hours"] for s in snaps])
    la = np.array([float(np.mean(s["lat"])) for s in snaps])
    lo = np.array([float(np.mean(s["lon"])) for s in snaps])
    return h, la, lo
