"""Land mask: stop the detector from marking dark land features (wetlands, rivers, car parks) as oil.

Uses Datasets/reference/land.geojson (Natural Earth land polygons, created by scripts/download_coastline.py).
Pure numpy + matplotlib, so it works without extra GIS libraries.
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from ml.config import LAND_PATH


def _rings(geom):
    t = geom.get("type")
    if t == "Polygon":
        return [geom["coordinates"][0]]
    if t == "MultiPolygon":
        return [poly[0] for poly in geom["coordinates"]]
    if t == "GeometryCollection":
        return [r for g in geom.get("geometries", []) for r in _rings(g)]
    return []


def land_mask(shape, transform, path=LAND_PATH, buffer_px=3):
    """Boolean HxW array, True on land (slightly widened so the shoreline itself is excluded)."""
    path = Path(path)
    h, w = shape
    mask = np.zeros(shape, bool)
    if not path.exists():
        return mask
    from matplotlib.path import Path as MplPath
    a, b, c, d, e, f = transform
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    lon = a * cols + b * rows + c
    lat = d * cols + e * rows + f
    pts = np.column_stack([lon.ravel(), lat.ravel()])
    lon_min, lon_max, lat_min, lat_max = lon.min(), lon.max(), lat.min(), lat.max()
    gj = json.loads(path.read_text())
    feats = gj["features"] if "features" in gj else [gj]
    for feat in feats:
        for ring in _rings(feat.get("geometry", feat) or {}):
            ring = np.asarray(ring, float)
            if ring[:, 0].max() < lon_min or ring[:, 0].min() > lon_max or ring[:, 1].max() < lat_min or ring[:, 1].min() > lat_max:
                continue
            mask |= MplPath(ring).contains_points(pts).reshape(shape)
    if buffer_px and mask.any():
        mask = ndimage.binary_dilation(mask, iterations=buffer_px)
    return mask
