"""Trained look-alike model (feature B4).

Learns "real oil slick" vs "look-alike" from the expert-labelled Sentinel-1 dataset of Yang & Singha (2025,
Eastern Mediterranean 2019, PANGAEA.980773, CC-BY-4.0): 3225 oil slicks in 1365 patches and 2290 patches with
look-alikes (low wind, rain cells, algae films, current fronts, ...).

The same measurements are made on a training sample and on a patch found by the segmentation models, so the
model can be used directly in the second-stage check:
  contrast, darkening, edge sharpness, texture (as in second_stage.py), plus shape (elongation, fill, compactness),
  size, background roughness (a stand-in for wind) and how dark the darkest part is.
All are ratios or scale-free numbers, because the training images (jpg, dB) and the app's images (8-bit DN) are
scaled differently. Training images (~20 m pixels) are reduced to ~80 m, and the app's crops are resampled
to the same ~80 m before measuring, so sizes and edges are compared at the same scale.
"""
import pickle
from pathlib import Path

import numpy as np
from scipy import ndimage

from ml.config import MODELS_DIR

MODEL_PATH = MODELS_DIR / "lookalike_learned.pkl"
RING_PX = 10
DOWNSAMPLE = 4            # dataset pixels are ~20 m; measurements are made at ~80 m
DATASET_PIXEL_M = 20.0
TARGET_PIXEL_M = DATASET_PIXEL_M * DOWNSAMPLE
FEATURES = ["contrast", "darkening", "edge_sharpness", "texture", "elongation", "fill", "compactness",
            "log_area_px", "background_roughness", "dark_core"]


def measure(img, mask):
    """Scale-free measurements of one dark patch. img: 2-D array (0 = no data), mask: bool, same shape."""
    img = img.astype(np.float32)
    valid = img > 0
    inside = mask & valid
    ring = ndimage.binary_dilation(mask, iterations=RING_PX) & ~mask & valid
    n = int(inside.sum())
    if n < 10 or ring.sum() < 10:
        return None
    m_in, m_out = float(img[inside].mean()), float(img[ring].mean())
    s_out = float(img[ring].std()) or 1.0
    smooth = ndimage.gaussian_filter(img, 1.5)
    grad = np.hypot(ndimage.sobel(smooth, 0), ndimage.sobel(smooth, 1))
    border = (mask ^ ndimage.binary_erosion(mask, iterations=2)) & valid
    g_border = float(grad[border].mean()) if border.any() else 0.0
    g_back = float(np.median(grad[ring])) or 1.0
    rr, cc = np.nonzero(mask)
    cov = np.cov(np.vstack([rr, cc])) if len(rr) > 2 else np.eye(2)
    ev = np.sort(np.linalg.eigvalsh(cov + 1e-6 * np.eye(2)))
    elong = float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
    box = (rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1)
    perim = float((mask ^ ndimage.binary_erosion(mask)).sum())
    return {"contrast": (m_out - m_in) / s_out,
            "darkening": 1 - m_in / max(m_out, 1e-6),
            "edge_sharpness": g_border / g_back,
            "texture": float(img[inside].std()) / s_out,
            "elongation": min(elong, 50.0),
            "fill": mask.sum() / box,
            "compactness": perim ** 2 / (4 * np.pi * mask.sum()),
            "log_area_px": float(np.log10(mask.sum())),
            "background_roughness": s_out / max(m_out, 1e-6),
            "dark_core": (m_out - float(np.percentile(img[inside], 10))) / s_out}


def vector(f):
    return np.array([[f[k] for k in FEATURES]], dtype=np.float64)


def dark_mask(img, k=1.0, min_px=20):
    """Dark-patch mask the way a detector would see it: clearly darker than the patch average."""
    img = img.astype(np.float32)
    valid = img > 0
    if valid.sum() < 50:
        return np.zeros(img.shape, bool)
    sm = ndimage.gaussian_filter(img, 1.0)
    mu, sd = float(sm[valid].mean()), float(sm[valid].std())
    m = (sm < mu - k * sd) & valid
    m = ndimage.binary_opening(m, iterations=1)
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.nonzero(sizes >= min_px)[0])


_MODEL = {"loaded": False, "obj": None}


def load(path=MODEL_PATH):
    if not _MODEL["loaded"]:
        _MODEL["loaded"] = True
        p = Path(path)
        if p.exists():
            try:
                with open(p, "rb") as fh:
                    _MODEL["obj"] = pickle.load(fh)
            except Exception:                            # noqa: BLE001 - the rule-based check still works
                _MODEL["obj"] = None
    return _MODEL["obj"]


def pixel_size_m(transform, lat):
    """Pixel size in metres of a lat/lon (EPSG:4326) affine transform."""
    a, b, _, d, e, _ = transform[:6]
    dx = np.hypot(a, d) * 111320 * np.cos(np.radians(lat))
    dy = np.hypot(b, e) * 110540
    return float(np.sqrt(dx * dy))


def to_target(img, mask, pixel_m):
    """Resample a crop to the pixel size the model was trained at (TARGET_PIXEL_M)."""
    if not pixel_m:
        return img, mask
    f = pixel_m / TARGET_PIXEL_M
    if 0.8 <= f <= 1.25:
        return img, mask
    valid = ndimage.zoom((img > 0).astype(np.float32), f, order=1) > 0.5
    out = ndimage.zoom(img.astype(np.float32), f, order=1)
    out[~valid] = 0
    m = ndimage.zoom(mask.astype(np.float32), f, order=1) > 0.5
    return out, m


def predict(img, mask, pixel_m=None, model=None):
    """{"p": probability of real oil, "features", "outside": features outside the training range} or None."""
    model = model or load()
    if model is None:
        return None
    img, mask = to_target(img, mask, pixel_m)
    f = measure(img, mask)
    if f is None:
        return None
    outside = []
    for k, (lo, hi) in (model.get("ranges") or {}).items():
        if k in f and not lo <= f[k] <= hi:
            outside.append(k)
    return {"p": float(model["model"].predict_proba(vector(f))[0, 1]),
            "features": {k: round(float(v), 3) for k, v in f.items()}, "outside": outside}
