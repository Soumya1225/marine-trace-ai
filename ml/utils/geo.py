"""Small, dependency-free geometry helpers (numpy only)."""
import numpy as np

KM_PER_DEG_LAT = 110.574


def km_per_deg_lon(lat):
    return 111.320 * np.cos(np.radians(lat))


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def to_local_km(lat, lon, lat0, lon0):
    """Equirectangular projection around (lat0, lon0) -> x east km, y north km."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    return (lon - lon0) * km_per_deg_lon(lat0), (lat - lat0) * KM_PER_DEG_LAT


def from_local_km(x, y, lat0, lon0):
    return lat0 + np.asarray(y) / KM_PER_DEG_LAT, lon0 + np.asarray(x) / km_per_deg_lon(lat0)


def bearing_deg(x, y):
    """Compass bearing (0 = north, clockwise) of a local km vector."""
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def axis_angle_diff(a, b):
    """Difference between two undirected axes in degrees (0..90)."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 180.0)
    return np.minimum(d, 180.0 - d)


def point_to_polyline_km(px, py, lx, ly):
    """Min distance from each point (px, py) to the polyline (lx, ly). All local km."""
    px, py = np.atleast_1d(px).astype(float), np.atleast_1d(py).astype(float)
    lx, ly = np.atleast_1d(lx).astype(float), np.atleast_1d(ly).astype(float)
    if len(lx) == 1:
        return np.hypot(px - lx[0], py - ly[0])
    ax, ay, bx, by = lx[:-1], ly[:-1], lx[1:], ly[1:]
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    seg_len2[seg_len2 == 0] = 1e-12
    t = ((px[:, None] - ax) * dx + (py[:, None] - ay) * dy) / seg_len2
    t = np.clip(t, 0, 1)
    cx, cy = ax + t * dx, ay + t * dy
    return np.min(np.hypot(px[:, None] - cx, py[:, None] - cy), axis=1)


def convex_hull_lonlat(lats, lons):
    """Convex hull ring [[lon, lat], ...] (closed). Falls back to a bounding box."""
    pts = np.column_stack([lons, lats])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return []
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        ring = pts[hull.vertices]
    except Exception:
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        ring = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    ring = np.vstack([ring, ring[:1]])
    return ring.round(6).tolist()


def apply_affine(transform, cols, rows):
    """transform = (a, b, c, d, e, f): lon = a*col + b*row + c ; lat = d*col + e*row + f."""
    a, b, c, d, e, f = transform
    cols, rows = np.asarray(cols, float), np.asarray(rows, float)
    return a * cols + b * rows + c, d * cols + e * rows + f


def transform_from_bbox(bbox, width, height):
    """bbox = (min_lon, min_lat, max_lon, max_lat) -> north-up affine tuple."""
    min_lon, min_lat, max_lon, max_lat = bbox
    return ((max_lon - min_lon) / width, 0.0, min_lon, 0.0, -(max_lat - min_lat) / height, max_lat)


def bounds_from_transform(transform, width, height):
    xs, ys = apply_affine(transform, [0, width, 0, width], [0, 0, height, height])
    return float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))
