"""Optical cross-check with Sentinel-2 (feature A6).

Radar sees oil as a dark, smooth patch, but so do look-alikes. A cloud-free optical image taken close in time
can confirm it: oil often shows as a sheen or a brown / silver streak. For an incident this module

  * searches the Copernicus Data Space catalogue (no login) for Sentinel-2 Level-2A images over the spill,
    a few days before and after the radar image, with their cloud cover;
  * fetches the small preview image (quicklook, about 343 x 343 px) once and caches it;
  * builds a Copernicus Browser link that opens the full-resolution true-colour image of that day.

Limits that the page states: Sentinel-2 sees only in daylight and not through cloud (the monsoon often hides
the sea), it passes over a place only every 2-5 days, and the preview covers a whole 110 km tile.
"""
import os
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from ml.config import RAW_DIR

CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
ASSET = "https://catalogue.dataspace.copernicus.eu/odata/v1/Assets({})/$value"
BROWSER = "https://browser.dataspace.copernicus.eu/"
CACHE = RAW_DIR / "sentinel2" / "quicklooks"
TIMEOUT_S = 60


def _iso(t):
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _attr(product, name):
    for a in product.get("Attributes") or []:
        if a.get("Name") == name:
            return a.get("Value")
    return None


def browser_link(lat, lon, when, zoom=11):
    day = when.astimezone(timezone.utc).strftime("%Y-%m-%d")
    q = {"zoom": zoom, "lat": round(lat, 4), "lng": round(lon, 4), "themeId": "DEFAULT-THEME",
         "datasetId": "S2_L2A_CDAS", "fromTime": f"{day}T00:00:00.000Z", "toTime": f"{day}T23:59:59.999Z",
         "layerId": "1_TRUE_COLOR"}
    return BROWSER + "?" + urllib.parse.urlencode(q)


def build_filter(lat, lon, when, days, max_cloud, pad_deg=0.05):
    lo0, la0, lo1, la1 = (round(v, 4) for v in (lon - pad_deg, lat - pad_deg, lon + pad_deg, lat + pad_deg))
    poly = f"POLYGON(({lo0} {la0}, {lo1} {la0}, {lo1} {la1}, {lo0} {la1}, {lo0} {la0}))"
    t0, t1 = when - timedelta(days=days), when + timedelta(days=days)
    return (f"Collection/Name eq 'SENTINEL-2' and "
            f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and "
            f"att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;{poly}') and "
            f"ContentDate/Start gt {_iso(t0)} and ContentDate/Start lt {_iso(t1)} and "
            f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and "
            f"att/OData.CSC.DoubleAttribute/Value le {max_cloud})")


def _query(flt, expand):
    # Same form as the Sentinel-1 downloader: the catalogue rejects "+" for spaces, so the filter is passed
    # inside a full URL and requests encodes spaces as %20.
    return f"{CATALOGUE}?$filter={flt}&$expand={expand}&$orderby=ContentDate/Start asc&$top=40"


def _assets(product_id, get):
    try:
        r = get(f"{CATALOGUE}({product_id})?$expand=Assets", timeout=TIMEOUT_S)
        r.raise_for_status()
        return r.json().get("Assets") or []
    except Exception:                                    # noqa: BLE001 - preview is optional
        return []


def search(lat, lon, when, days=3, max_cloud=90, get=requests.get):
    """Sentinel-2 L2A images over (lat, lon) within +-days of `when`, nearest in time first."""
    flt = build_filter(lat, lon, when, days, max_cloud)
    r = get(_query(flt, "Attributes,Assets"), timeout=TIMEOUT_S)
    with_assets = r.status_code < 400
    if not with_assets:                                  # some catalogue versions accept only one $expand
        r = get(_query(flt, "Attributes"), timeout=TIMEOUT_S)
    r.raise_for_status()
    seen, out = set(), []
    for p in r.json().get("value", []):
        start = datetime.fromisoformat(p["ContentDate"]["Start"].replace("Z", "+00:00"))
        tile = p["Name"].split("_")[5] if p["Name"].count("_") >= 5 else p["Name"]
        key = (start.strftime("%Y%m%dT%H"), tile)
        if key in seen:                                  # the same image is sometimes listed twice (reprocessing)
            continue
        seen.add(key)
        assets = p.get("Assets") if with_assets else None
        if assets is None and len(out) < 12 and p.get("Id"):
            assets = _assets(p["Id"], get)
        ql = next((a for a in assets or [] if str(a.get("Type", "")).upper() == "QUICKLOOK"), None)
        cloud = _attr(p, "cloudCover")
        out.append({"name": p["Name"], "time": start.isoformat().replace("+00:00", "Z"), "tile": tile,
                    "cloud_pct": round(float(cloud), 1) if cloud is not None else None,
                    "hours_from_radar": round((start - when).total_seconds() / 3600, 1),
                    "quicklook_id": ql.get("Id") if ql else None,
                    "browser_url": browser_link(lat, lon, start)})
    out.sort(key=lambda s: abs(s["hours_from_radar"]))
    return out


def quicklook(asset_id, get=requests.get):
    """JPEG bytes of a quicklook, cached on disk. Tries without login first, then with the CDSE token."""
    if not asset_id or not all(c.isalnum() or c == "-" for c in asset_id):
        raise ValueError("Invalid preview id")
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{asset_id}.jpg"
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    url = ASSET.format(asset_id)
    r = get(url, timeout=TIMEOUT_S)
    if r.status_code in (401, 403) and os.getenv("CDSE_USER"):
        from ml.preprocessing.download_sentinel1 import get_token
        r = get(url, timeout=TIMEOUT_S, headers={"Authorization": f"Bearer {get_token()}"})
    r.raise_for_status()
    path.write_bytes(r.content)
    return r.content


def for_incident(inc, days=3, max_cloud=90, get=requests.get):
    when = inc.detected_at or inc.created_at
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    scenes = search(inc.centroid_lat, inc.centroid_lon, when, days, max_cloud, get=get)
    usable = [s for s in scenes if s["cloud_pct"] is not None and s["cloud_pct"] <= 30]
    if not scenes:
        note = (f"No Sentinel-2 image within {days} days of the radar image. Optical satellites pass every few days "
                "and only in daylight.")
    elif not usable:
        note = ("Sentinel-2 images exist but all are very cloudy (over 30%): the oil probably cannot be seen. "
                "This is common in the monsoon.")
    else:
        best = usable[0]
        note = (f"Best image: {best['hours_from_radar']:+.0f} h from the radar image, {best['cloud_pct']}% cloud. "
                "Open it and look for a sheen or a brown / silver streak at the spill position.")
    return {"radar_time": when.isoformat().replace("+00:00", "Z"), "lat": inc.centroid_lat, "lon": inc.centroid_lon,
            "days": days, "scenes": scenes, "note": note}
