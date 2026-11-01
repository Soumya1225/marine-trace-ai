"""Turn a predicted class map into individual spills with geographic features."""
import numpy as np
from scipy import ndimage

from ml.config import MIN_SPILL_PIXELS, OIL_CLASS
from ml.utils.geo import (KM_PER_DEG_LAT, apply_affine, bearing_deg, convex_hull_lonlat,
                          from_local_km, km_per_deg_lon, to_local_km)


def _polygon_rings(region_mask, transform, offset):
    """Exact outline with rasterio if available, otherwise convex hull."""
    try:
        from rasterio.features import shapes
        from affine import Affine
        a, b, c, d, e, f = transform
        r0, c0 = offset
        aff = Affine(a, b, c, d, e, f) @ Affine.translation(c0, r0)
        polys = [geom for geom, val in shapes(region_mask.astype(np.uint8), mask=region_mask, transform=aff) if val == 1]
        if polys:
            biggest = max(polys, key=lambda g: len(g["coordinates"][0]))
            ring = np.array(biggest["coordinates"][0])
            step = max(1, len(ring) // 200)  # keep polygons light for the web map
            ring = np.vstack([ring[::step], ring[:1]])
            return ring.round(6).tolist()
    except Exception:
        pass
    rows, cols = np.nonzero(region_mask)
    lon, lat = apply_affine(transform, cols + offset[1] + 0.5, rows + offset[0] + 0.5)
    return convex_hull_lonlat(lat, lon)


def describe_points(lat, lon):
    """Shape statistics for a cloud of spill pixels / particles (local km PCA)."""
    lat0, lon0 = float(np.mean(lat)), float(np.mean(lon))
    x, y = to_local_km(lat, lon, lat0, lon0)
    if len(x) < 3:
        return {"centroid_lat": lat0, "centroid_lon": lon0, "orientation_deg": 0.0,
                "length_km": 0.0, "width_km": 0.0, "elongation": 1.0,
                "axis": [[lon0, lat0]]}
    cov = np.cov(np.vstack([x, y]))
    vals, vecs = np.linalg.eigh(cov)
    major = vecs[:, np.argmax(vals)]
    minor = vecs[:, np.argmin(vals)]
    proj_major = x * major[0] + y * major[1]
    proj_minor = x * minor[0] + y * minor[1]
    length = float(proj_major.max() - proj_major.min())
    width = float(max(proj_minor.max() - proj_minor.min(), 0.01))
    t = np.linspace(proj_major.min(), proj_major.max(), 25)
    ax_lat, ax_lon = from_local_km(t * major[0], t * major[1], lat0, lon0)
    return {
        "centroid_lat": lat0,
        "centroid_lon": lon0,
        "orientation_deg": float(bearing_deg(major[0], major[1]) % 180.0),
        "length_km": round(length, 3),
        "width_km": round(width, 3),
        "elongation": round(length / width, 2),
        "axis": np.column_stack([ax_lon, ax_lat]).round(6).tolist(),
    }


def extract_spills(class_map, transform, acquisition_time=None, oil_prob=None,
                   oil_class=OIL_CLASS, min_pixels=MIN_SPILL_PIXELS, uncertainty=None):
    """Return a list of spill dicts sorted by area (largest first)."""
    labels, n = ndimage.label(class_map == oil_class)
    if n == 0:
        return []
    a, b, _, d, e, _ = transform
    spills = []
    for idx, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        region = labels[sl] == idx
        npx = int(region.sum())
        if npx < min_pixels:
            continue
        rows, cols = np.nonzero(region)
        rows = rows + sl[0].start
        cols = cols + sl[1].start
        lon, lat = apply_affine(transform, cols + 0.5, rows + 0.5)
        lat_c = float(np.mean(lat))
        # pixel footprint in km^2 from the affine determinant
        px_area_km2 = abs(a * e - b * d) * km_per_deg_lon(lat_c) * KM_PER_DEG_LAT
        shape = describe_points(lat, lon)
        conf = float(np.mean(oil_prob[labels == idx])) if oil_prob is not None else None
        unc = float(np.mean(uncertainty[labels == idx])) if uncertainty is not None else None
        spills.append({
            **shape,
            "area_km2": round(npx * px_area_km2, 4),
            "pixels": npx,
            "confidence": round(conf, 3) if conf is not None else None,
            "model_disagreement": round(unc, 3) if unc is not None else None,
            "polygon": _polygon_rings(region, transform, (sl[0].start, sl[1].start)),
            "acquisition_time": acquisition_time.isoformat() if acquisition_time else None,
        })
    spills.sort(key=lambda s: s["area_km2"], reverse=True)
    for i, s in enumerate(spills, start=1):
        s["spill_no"] = i
    return spills
