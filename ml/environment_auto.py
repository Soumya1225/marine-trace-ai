"""Automatic wind and ocean-current data for every analysis (feature A5).

Before, the drift model used whatever NetCDF file sat in Datasets/raw/weather, even when that file covered a
different sea (a Huntington Beach file used for a Mumbai image gives California's wind). Now, for each analysis:

  1. a file that really covers the image area and the forecast period (image time - 1 day ... + 3 days) is used;
  2. if there is none, it is downloaded once and kept in Datasets/raw/weather/auto/ for next time:
       wind     - Open-Meteo historical API (ERA5 / ECMWF reanalysis, 1940 onwards, free, no account);
                  if that fails and ~/.cdsapirc exists: ERA5 directly from the Copernicus Climate Data Store;
       currents - Copernicus Marine (GLORYS12 reanalysis) when `copernicusmarine login` has been run,
                  otherwise Open-Meteo marine API (Meteo-France SMOC, available from January 2022);
  3. if nothing can be downloaded (offline, too old, no account), the default values are used and the analysis
     records why, so the incident page can say "default current used".

Values typed by the analyst on the New analysis page always win (no download for that variable).
"""
import json
import os
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ml.config import RAW_DIR

WEATHER = RAW_DIR / "weather"
AUTO = WEATHER / "auto"
PAD_DEG = 0.25                       # download a little more than the image area
DAYS_BEFORE, DAYS_AFTER = 1, 3       # back-tracking (12 h) and the 72 h forecast
TIMEOUT_S = 60
OPENMETEO_WIND = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_MARINE = "https://marine-api.open-meteo.com/v1/marine"
MARINE_FROM = date(2022, 1, 1)       # Open-Meteo marine currents start here
CMEMS_DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m"


# ------------------------------------------------------------------ helpers
def period(when):
    t = when.astimezone(timezone.utc) if when.tzinfo else when.replace(tzinfo=timezone.utc)
    return (t - timedelta(days=DAYS_BEFORE)).date(), (t + timedelta(days=DAYS_AFTER)).date()


def padded(bbox):
    lo0, la0, lo1, la1 = bbox
    return (round(lo0 - PAD_DEG, 2), round(la0 - PAD_DEG, 2), round(lo1 + PAD_DEG, 2), round(la1 + PAD_DEG, 2))


def auto_name(kind, source, bbox, d0, d1):
    lo0, la0, lo1, la1 = padded(bbox)
    return f"{kind}_{source}_{d0:%Y%m%d}-{d1:%Y%m%d}_{la0:+.2f}_{lo0:+.2f}_{la1:+.2f}_{lo1:+.2f}.nc"


def grid_points(bbox, n=3):
    """n x n points over the (padded) area: Open-Meteo answers one time series per point."""
    lo0, la0, lo1, la1 = padded(bbox)
    return np.linspace(la0, la1, n).round(3), np.linspace(lo0, lo1, n).round(3)


def _get_json(url, params):
    full = url + "?" + urllib.parse.urlencode(params, safe=",")
    with urllib.request.urlopen(full, timeout=TIMEOUT_S) as r:          # noqa: S310 - fixed https hosts
        return json.loads(r.read().decode("utf-8"))


# ------------------------------------------------------------------ is an existing file good enough?
_COVER_CACHE = {}


def coverage(path):
    """(min_lat, max_lat, min_lon, max_lon, t_first, t_last) of a NetCDF file, or None if unreadable."""
    key = (str(path), Path(path).stat().st_mtime)
    if key in _COVER_CACHE:
        return _COVER_CACHE[key]
    cov = None
    try:
        import xarray as xr
        with xr.open_dataset(path) as ds:
            lat = ds["latitude" if "latitude" in ds.coords else "lat"].values
            lon = ds["longitude" if "longitude" in ds.coords else "lon"].values
            tname = "valid_time" if "valid_time" in ds.coords else "time"
            t = ds[tname].values
            cov = (float(np.min(lat)), float(np.max(lat)), float(np.min(lon)), float(np.max(lon)),
                   np.datetime64(np.min(t), "s").astype(datetime), np.datetime64(np.max(t), "s").astype(datetime))
    except Exception:                                                  # noqa: BLE001
        cov = None
    _COVER_CACHE[key] = cov
    return cov


def covers(cov, bbox, when, tol_deg=0.3, tol_h=24):
    """True if the file covers the image centre and the image time (with a little tolerance)."""
    if cov is None:
        return False
    la0, la1, lo0, lo1, t0, t1 = cov
    clat, clon = (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2
    t = when.astimezone(timezone.utc).replace(tzinfo=None) if when.tzinfo else when
    inside = (la0 - tol_deg <= clat <= la1 + tol_deg) and (lo0 - tol_deg <= clon <= lo1 + tol_deg)
    return inside and (t0 - timedelta(hours=tol_h) <= t <= t1 + timedelta(hours=tol_h))


def find_covering(kind, bbox, when, cov_fn=coverage):
    pats = ["*wind*.nc", "era5*.nc"] if kind == "wind" else ["*current*.nc", "cmems*.nc"]
    files = []
    for folder in (AUTO, WEATHER):
        if folder.exists():
            for p in pats:
                files += sorted(folder.glob(p))
    for f in dict.fromkeys(files):
        if covers(cov_fn(f), bbox, when):
            return f
    return None


# ------------------------------------------------------------------ Open-Meteo (no account)
def _uv_from(speed, direction_deg, towards):
    """u (east) / v (north) from speed and compass direction. Wind direction is where it comes FROM,
    current direction is where it flows TOWARDS."""
    s = np.asarray(speed, float)
    d = np.radians(np.asarray(direction_deg, float))
    sign = 1.0 if towards else -1.0
    return sign * s * np.sin(d), sign * s * np.cos(d)


def openmeteo_grid(kind, bbox, d0, d1, fetch=_get_json, n=3):
    """Download an n x n grid of hourly series. Returns (times, lats, lons, u[t, lat, lon], v[t, lat, lon])."""
    lats, lons = grid_points(bbox, n)
    pts = [(la, lo) for la in lats for lo in lons]
    params = {"latitude": ",".join(str(p[0]) for p in pts), "longitude": ",".join(str(p[1]) for p in pts),
              "start_date": d0.isoformat(), "end_date": d1.isoformat(), "timezone": "GMT"}
    if kind == "wind":
        params.update(hourly="wind_speed_10m,wind_direction_10m", wind_speed_unit="ms")
        url, sp_key, dir_key, towards = OPENMETEO_WIND, "wind_speed_10m", "wind_direction_10m", False
    else:
        params.update(hourly="ocean_current_velocity,ocean_current_direction", cell_selection="sea")
        url, sp_key, dir_key, towards = OPENMETEO_MARINE, "ocean_current_velocity", "ocean_current_direction", True
    res = fetch(url, params)
    res = res if isinstance(res, list) else [res]
    if len(res) != len(pts):
        raise ValueError(f"Open-Meteo returned {len(res)} locations instead of {len(pts)}")
    times = np.array(res[0]["hourly"]["time"], dtype="datetime64[s]")
    u = np.full((len(times), len(lats), len(lons)), np.nan)
    v = np.full_like(u, np.nan)
    for k, r in enumerate(res):
        h = r["hourly"]
        speed = np.array([np.nan if x is None else x for x in h[sp_key]], float)
        unit = (r.get("hourly_units") or {}).get(sp_key, "m/s")
        if unit.replace(" ", "").lower() in ("km/h", "kmh"):
            speed = speed / 3.6
        elif unit.lower() in ("kn", "knots"):
            speed = speed * 0.514444
        direc = np.array([np.nan if x is None else x for x in h[dir_key]], float)
        uu, vv = _uv_from(speed, direc, towards)
        u[:, k // len(lons), k % len(lons)] = uu
        v[:, k // len(lons), k % len(lons)] = vv
    if not np.isfinite(u).any():
        raise ValueError("Open-Meteo returned no values for this area and period")
    return times, lats, lons, u, v


def write_nc(path, kind, times, lats, lons, u, v, source):
    """Save in the same layout as ERA5 / CMEMS files, so drift.Environment reads it unchanged."""
    import xarray as xr
    un, vn = ("u10", "v10") if kind == "wind" else ("uo", "vo")
    ds = xr.Dataset({un: (("time", "latitude", "longitude"), u.astype("float32")),
                     vn: (("time", "latitude", "longitude"), v.astype("float32"))},
                    coords={"time": times, "latitude": lats, "longitude": lons},
                    attrs={"source": source, "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


# ------------------------------------------------------------------ Copernicus (accounts)
def has_cds():
    if not (Path.home() / ".cdsapirc").exists() and not os.getenv("CDSAPI_KEY"):
        return False
    try:
        import cdsapi  # noqa: F401
        return True
    except ImportError:
        return False


def has_cmems():
    try:
        import copernicusmarine  # noqa: F401
    except ImportError:
        return False
    return bool(os.getenv("COPERNICUSMARINE_SERVICE_USERNAME")) or any(
        (Path.home() / ".copernicusmarine").glob("*credentials*"))


def era5_cds(bbox, d0, d1, out):
    import cdsapi
    lo0, la0, lo1, la1 = padded(bbox)
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    cdsapi.Client(quiet=True).retrieve("reanalysis-era5-single-levels", {
        "product_type": ["reanalysis"], "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        "year": sorted({d.strftime("%Y") for d in days}), "month": sorted({d.strftime("%m") for d in days}),
        "day": sorted({d.strftime("%d") for d in days}), "time": [f"{h:02d}:00" for h in range(24)],
        "area": [la1, lo0, la0, lo1], "data_format": "netcdf"}, str(out))
    return out


def cmems_currents(bbox, d0, d1, out):
    import copernicusmarine
    lo0, la0, lo1, la1 = padded(bbox)
    copernicusmarine.subset(dataset_id=CMEMS_DATASET, variables=["uo", "vo"],
                            minimum_longitude=lo0, maximum_longitude=lo1, minimum_latitude=la0, maximum_latitude=la1,
                            start_datetime=f"{d0}T00:00:00", end_datetime=f"{d1}T23:59:59",
                            minimum_depth=0.0, maximum_depth=1.0,
                            output_directory=str(out.parent), output_filename=out.name)
    return out


# ------------------------------------------------------------------ main entry
def _wind(bbox, when, log):
    d0, d1 = period(when)
    try:
        times, lats, lons, u, v = openmeteo_grid("wind", bbox, d0, d1)
        out = write_nc(AUTO / auto_name("wind", "openmeteo", bbox, d0, d1), "wind", times, lats, lons, u, v,
                       "Open-Meteo historical API (ERA5 / ECMWF reanalysis)")
        return out, "wind downloaded from Open-Meteo (ERA5 reanalysis)"
    except Exception as exc:                                           # noqa: BLE001
        log(f"Open-Meteo wind failed: {exc}")
    if has_cds():
        try:
            out = era5_cds(bbox, d0, d1, AUTO / auto_name("wind", "era5", bbox, d0, d1))
            return out, "wind downloaded from the Copernicus Climate Data Store (ERA5)"
        except Exception as exc:                                       # noqa: BLE001
            log(f"ERA5 download failed: {exc}")
    return None, "wind could not be downloaded: default wind used"


def _current(bbox, when, log):
    d0, d1 = period(when)
    if has_cmems():
        try:
            out = cmems_currents(bbox, d0, d1, AUTO / auto_name("current", "cmems", bbox, d0, d1))
            return out, "currents downloaded from Copernicus Marine (GLORYS12 reanalysis)"
        except Exception as exc:                                       # noqa: BLE001
            log(f"Copernicus Marine download failed: {exc}")
    if d1 >= MARINE_FROM:
        try:
            times, lats, lons, u, v = openmeteo_grid("current", bbox, max(d0, MARINE_FROM), d1)
            out = write_nc(AUTO / auto_name("current", "openmeteo", bbox, d0, d1), "current", times, lats, lons, u, v,
                           "Open-Meteo marine API (Meteo-France SMOC)")
            return out, "currents downloaded from Open-Meteo (Meteo-France SMOC)"
        except Exception as exc:                                       # noqa: BLE001
            log(f"Open-Meteo currents failed: {exc}")
        return None, "currents could not be downloaded: default current used"
    return None, ("no free current data before 2022 and no Copernicus Marine login: default current used "
                  "(run `copernicusmarine login` once to enable it)")


def prepare(bbox, when, need_wind=True, need_current=True, log=print):
    """Return {"wind_file", "current_file", "notes": [...]} for this image area and time."""
    out = {"wind_file": None, "current_file": None, "notes": []}
    if bbox is None or when is None:
        return out
    for kind, need, fetch in (("wind", need_wind, _wind), ("current", need_current, _current)):
        if not need:
            out["notes"].append(f"{kind}: value typed by the analyst")
            continue
        found = find_covering(kind, bbox, when)
        if found is not None:
            out[f"{kind}_file"] = found
            out["notes"].append(f"{kind}: {found.name} (already downloaded)")
            continue
        path, note = fetch(bbox, when, log)
        out[f"{kind}_file"] = path
        out["notes"].append(f"{kind}: {note}")
    return out
