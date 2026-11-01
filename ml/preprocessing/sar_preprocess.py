"""Load a SAR scene, crop it to an area, reduce speckle and convert it to an 8-bit image for the model.

Supported inputs:
  * Sentinel-1 GRD .SAFE zip (normal or _COG) downloaded from Copernicus (reads the VV band)
  * A georeferenced GeoTIFF
  * A plain PNG/JPG (then a bbox must be given)

Radiometric calibration: for a Sentinel-1 SAFE zip the pixels are also converted to sigma0 (radar backscatter
per unit area) with ESA's calibration table (annotation/calibration/calibration-*.xml) and the thermal noise
is removed with the noise table (noise-*.xml):  sigma0 = (DN^2 - noise) / A_sigma^2   (ESA S1 calibration note).
sigma0 is returned as scene["sigma0_db"] (one physical scale across dates and satellites) and used for the
measured backscatter values. The detection models still get the DN image (10*log10 DN^2), the look they were
trained on: feeding them sigma0 changed their output (more weak false alarms near ports in the test scene).
"""
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage

from ml.config import MAX_SCENE_SIZE
from ml.utils.geo import apply_affine, bounds_from_transform, transform_from_bbox

IMAGE_EXT = {".png", ".jpg", ".jpeg"}


def parse_acquisition_time(name):
    m = re.search(r"(\d{8}T\d{6})", str(name))
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def _find_vv_in_zip(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        tiffs = [n for n in z.namelist() if n.lower().endswith((".tif", ".tiff")) and "measurement" in n.lower()]
    if not tiffs:
        raise ValueError("No measurement GeoTIFF found inside the SAFE zip")
    vv = [n for n in tiffs if "-vv-" in n.lower()] or tiffs
    return vv[0]


# ------------------------------------------------------------------ radiometric calibration (Sentinel-1 GRD)
def _annotation_names(zip_names, tiff_name):
    """Calibration and noise XML that belong to one measurement GeoTIFF (same file stem)."""
    stem = Path(tiff_name).stem.lower()                  # e.g. s1a-iw-grd-vv-20211003t015016-...-001
    stem = re.sub(r"[-_]cog$", "", stem)                  # COG products add "-cog" to the image name only
    cal = [n for n in zip_names if "/calibration/calibration-" in n.lower() and stem in n.lower()]
    noise = [n for n in zip_names if "/calibration/noise-" in n.lower() and stem in n.lower()]
    return (cal[0] if cal else None), (noise[0] if noise else None)


def _floats(text):
    return np.array(text.split(), dtype=np.float64) if text and text.strip() else np.array([], np.float64)


def parse_calibration(xml_text, lut="sigmaNought"):
    """[(line, pixels, values)] from a calibration XML (one vector every few hundred lines)."""
    root = ET.fromstring(xml_text)
    out = []
    for v in root.iter("calibrationVector"):
        out.append((float(v.findtext("line")), _floats(v.findtext("pixel")), _floats(v.findtext(lut))))
    if not out:
        raise ValueError("No calibration vectors in the calibration XML")
    return out


def parse_noise(xml_text):
    """Range noise vectors and azimuth noise blocks from a noise XML (old and new IPF layouts)."""
    root = ET.fromstring(xml_text)
    rng = [(float(v.findtext("line")), _floats(v.findtext("pixel")), _floats(v.findtext("noiseRangeLut")))
           for v in root.iter("noiseRangeVector")]
    if not rng:                                            # IPF < 2.9: one combined noise LUT
        rng = [(float(v.findtext("line")), _floats(v.findtext("pixel")), _floats(v.findtext("noiseLut")))
               for v in root.iter("noiseVector")]
    az = []
    for v in root.iter("noiseAzimuthVector"):
        az.append({"first_col": int(float(v.findtext("firstRangeSample") or 0)),
                   "last_col": int(float(v.findtext("lastRangeSample") or 1e9)),
                   "first_row": int(float(v.findtext("firstAzimuthLine") or 0)),
                   "last_row": int(float(v.findtext("lastAzimuthLine") or 1e9)),
                   "lines": _floats(v.findtext("line")), "values": _floats(v.findtext("noiseAzimuthLut"))})
    return rng, az


def lut_grid(vectors, rows, cols):
    """Bilinear interpolation of sparse LUT vectors [(line, pixels, values)] onto a grid of rows x cols."""
    vectors = sorted(vectors, key=lambda v: v[0])
    lines = np.array([v[0] for v in vectors])
    along = np.vstack([np.interp(cols, px, val) for _, px, val in vectors])     # (K, W)
    if len(vectors) == 1:
        return np.repeat(along, len(rows), axis=0)
    r = np.clip(rows, lines[0], lines[-1])
    i = np.clip(np.searchsorted(lines, r, side="right") - 1, 0, len(lines) - 2)
    t = ((r - lines[i]) / np.maximum(lines[i + 1] - lines[i], 1e-9))[:, None]
    return along[i] * (1 - t) + along[i + 1] * t


def azimuth_noise_grid(blocks, rows, cols):
    """Azimuth noise factor per pixel (1 where no block applies)."""
    out = np.ones((len(rows), len(cols)))
    for b in blocks:
        cm = (cols >= b["first_col"]) & (cols <= b["last_col"])
        rm = (rows >= b["first_row"]) & (rows <= b["last_row"])
        if not cm.any() or not rm.any() or len(b["values"]) == 0:
            continue
        if len(b["values"]) == 1 or len(b["lines"]) < 2:
            vals = np.full(rm.sum(), b["values"][0])
        else:
            vals = np.interp(rows[rm], b["lines"], b["values"])
        out[np.ix_(rm, cm)] = vals[:, None]
    return out


def calibrate_sigma0(dn, rows, cols, cal_vectors, noise=None):
    """sigma0 (linear) for DN values sampled at the given full-resolution row / column positions."""
    a = lut_grid(cal_vectors, rows, cols)
    power = dn.astype(np.float64) ** 2
    if noise is not None:
        rng, az = noise
        if rng:
            n = lut_grid(rng, rows, cols)
            if az:
                n = n * azimuth_noise_grid(az, rows, cols)
            power = power - n
    sigma0 = power / np.maximum(a, 1e-9) ** 2
    sigma0 = np.maximum(sigma0, 1e-5)                     # noise floor: -50 dB, never zero or negative
    sigma0[dn <= 0] = 0                                    # keep no-data as 0 (masked later)
    return sigma0.astype(np.float32)


def lee_filter(img, size=5):
    """Classic Lee speckle filter."""
    img = img.astype(np.float32)
    mean = ndimage.uniform_filter(img, size)
    sq_mean = ndimage.uniform_filter(img * img, size)
    var = np.maximum(sq_mean - mean * mean, 0)
    noise = np.nanmean(var)
    weight = var / (var + noise + 1e-12)
    return mean + weight * (img - mean)


def to_uint8(db):
    valid = np.isfinite(db)
    if not valid.any():
        return np.zeros(db.shape, np.uint8)
    lo, hi = np.percentile(db[valid], [2, 98])
    # valid pixels -> 1..255, no-data -> 0
    out = 1 + np.clip((db - lo) / max(hi - lo, 1e-6), 0, 1) * 254
    out[~valid] = 0
    return out.astype(np.uint8)


def _read_raster(path, bbox, max_size):
    import rasterio
    from rasterio.control import GroundControlPoint
    from rasterio.transform import Affine, from_gcps
    from rasterio.warp import Resampling, calculate_default_transform, reproject
    from rasterio.windows import Window

    path = Path(path)
    open_path = str(path)
    meta = {"calibration": "approximate (10*log10 DN^2)"}
    cal_vectors = noise = None
    if path.suffix.lower() == ".zip":
        member = _find_vv_in_zip(path)
        open_path = f"zip://{path.resolve().as_posix()}!/{member}"
        try:
            with zipfile.ZipFile(path) as z:
                cal_name, noise_name = _annotation_names(z.namelist(), member)
                if cal_name:
                    cal_vectors = parse_calibration(z.read(cal_name).decode("utf-8", "ignore"))
                if cal_name and noise_name:
                    noise = parse_noise(z.read(noise_name).decode("utf-8", "ignore"))
        except Exception as exc:                          # noqa: BLE001 - fall back to the approximate scale
            cal_vectors = noise = None
            meta["calibration_error"] = str(exc)[:200]

    with rasterio.open(open_path) as src:
        georeferenced = src.crs is not None and not src.transform.is_identity
        if georeferenced:
            if src.crs.to_epsg() != 4326:
                from rasterio.vrt import WarpedVRT
                src = WarpedVRT(src, crs="EPSG:4326")
            aff = src.transform
            gcps, gcp_crs = None, None
        else:
            gcps, gcp_crs = src.gcps
            if not gcps:
                raise ValueError("Raster has neither a geotransform nor GCPs")
            aff = from_gcps(gcps)     # approximate, only used to find the pixel window

        width, height = src.width, src.height
        col0, row0, col1, row1 = 0, 0, width, height
        if bbox is not None:
            pad = 0.05  # degrees, so the window still covers the box after warping
            inv = ~aff
            xs = [bbox[0] - pad, bbox[2] + pad, bbox[0] - pad, bbox[2] + pad]
            ys = [bbox[1] - pad, bbox[1] - pad, bbox[3] + pad, bbox[3] + pad]
            cols, rows = zip(*[inv * (x, y) for x, y in zip(xs, ys)])
            col0, col1 = int(max(0, np.floor(min(cols)))), int(min(width, np.ceil(max(cols))))
            row0, row1 = int(max(0, np.floor(min(rows)))), int(min(height, np.ceil(max(rows))))
            if col1 <= col0 or row1 <= row0:
                raise ValueError("The requested area is outside this scene")
        win_w, win_h = col1 - col0, row1 - row0
        scale = max(1.0, max(win_w, win_h) / max_size)
        out_w, out_h = int(win_w / scale), int(win_h / scale)
        data = src.read(1, window=Window(col0, row0, win_w, win_h), out_shape=(out_h, out_w),
                        resampling=Resampling.average).astype(np.float32)
        sx, sy = win_w / out_w, win_h / out_h

    sigma0 = None
    if cal_vectors is not None:
        # full-resolution position of every (averaged) output pixel centre, where the LUTs are defined
        rows = row0 + (np.arange(out_h) + 0.5) * sy - 0.5
        cols = col0 + (np.arange(out_w) + 0.5) * sx - 0.5
        sigma0 = calibrate_sigma0(data, rows, cols, cal_vectors, noise)
        meta["calibration"] = "sigma0 (ESA calibration LUT" + (" + thermal noise removal)" if noise else ")")
        # the detection models were trained on the uncalibrated look, so they keep getting the DN image;
        # sigma0 travels along as a second band and is used for physical values only
        data = np.stack([data, sigma0])

    if georeferenced:
        dst_transform = aff @ Affine.translation(col0, row0) @ Affine.scale(sx, sy)
    else:
        # Sentinel-1 GRD is in radar geometry (tilted, mirrored on some passes):
        # warp it onto a north-up lat/lon grid using the satellite's ground control points.
        local = [GroundControlPoint(row=(g.row - row0) / sy, col=(g.col - col0) / sx, x=g.x, y=g.y, z=g.z)
                 for g in gcps]
        dst_transform, dst_w, dst_h = calculate_default_transform(gcp_crs, "EPSG:4326", out_w, out_h, gcps=local)
        warped = np.full(data.shape[:-2] + (dst_h, dst_w), np.nan, np.float32)
        reproject(data, warped, gcps=local, src_crs=gcp_crs, src_nodata=0,
                  dst_transform=dst_transform, dst_crs="EPSG:4326", dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
        data = warped

    if bbox is not None:   # crop the north-up grid exactly to the requested box
        inv = ~dst_transform
        c_a, r_a = inv * (bbox[0], bbox[3])
        c_b, r_b = inv * (bbox[2], bbox[1])
        c0, c1 = int(max(0, np.floor(min(c_a, c_b)))), int(min(data.shape[-1], np.ceil(max(c_a, c_b))))
        r0, r1 = int(max(0, np.floor(min(r_a, r_b)))), int(min(data.shape[-2], np.ceil(max(r_a, r_b))))
        if c1 > c0 and r1 > r0:
            data = data[..., r0:r1, c0:c1]
            dst_transform = dst_transform @ Affine.translation(c0, r0)

    if data.ndim == 3:                                      # [DN, sigma0] -> DN image + sigma0 in meta
        meta["sigma0"] = data[1]
        data = data[0]
    t = dst_transform
    return data, (t.a, t.b, t.c, t.d, t.e, t.f), meta


def load_scene(path, bbox=None, acquisition_time=None, max_size=MAX_SCENE_SIZE):
    """Return dict(image_u8, db, transform, bounds, acquisition_time, source, calibration, sigma0_db)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXT:
        if bbox is None:
            raise ValueError("A bbox is required for PNG/JPG images (no coordinates inside)")
        from PIL import Image
        img = Image.open(path).convert("L")
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size))
        data = np.asarray(img, dtype=np.float32)
        transform = transform_from_bbox(bbox, data.shape[1], data.shape[0])
        db = data  # already an image: keep values
        is_db = True
        meta = {"calibration": "none (plain image)"}
    else:
        data, transform, meta = _read_raster(path, bbox, max_size)
        data[~(data > 0)] = np.nan
        db = 10.0 * np.log10(data ** 2)                   # image for the models: DN amplitude -> power (dB)
        is_db = False

    if not is_db or suffix in IMAGE_EXT:
        db = np.where(np.isfinite(db), lee_filter(np.nan_to_num(db, nan=np.nanmedian(db))), np.nan)

    acq = acquisition_time or parse_acquisition_time(path.name)
    if isinstance(acq, str):
        acq = datetime.fromisoformat(acq.replace("Z", "+00:00"))
    if acq is not None and acq.tzinfo is None:
        acq = acq.replace(tzinfo=timezone.utc)

    sigma0_db = None
    if meta.get("sigma0") is not None:
        s0 = meta["sigma0"].astype(np.float32)
        s0[~(s0 > 0)] = np.nan
        sigma0_db = (10.0 * np.log10(s0)).astype(np.float32)   # calibrated backscatter (dB), physical scale

    h, w = db.shape
    return {
        "image_u8": to_uint8(db),
        "db": db.astype(np.float32),
        "transform": transform,
        "bounds": bounds_from_transform(transform, w, h),
        "acquisition_time": acq,
        "source": path.name,
        "calibration": meta.get("calibration"),
        "sigma0_db": sigma0_db,
    }


def pixel_lonlat(scene, cols, rows):
    return apply_affine(scene["transform"], cols, rows)
