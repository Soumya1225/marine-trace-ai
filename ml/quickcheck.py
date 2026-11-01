"""Quick image check: give any single picture, get a verdict (oil / no oil / possible look-alike / not a radar image).

Step 1, gate: the detection models only understand satellite radar (SAR) images of the sea. A radar image is
grey (no colour) and grainy everywhere (speckle), so it has almost no perfectly smooth areas. Ordinary photos
are coloured, or have large smooth areas (sky, walls, skin). Images that fail the gate are rejected instead of
being run through the model, so a random photo never gets "oil" painted on it.

Step 2, verdict: the detector (deep ensemble) runs on the image and the result is summarised:
  * no oil            - oil covers less than MIN_OIL_FRACTION of the image
  * possible look-alike - the ensemble models disagree or the model is unsure (calm water, algae and rain
                          cells look dark too; full analyses also use the wind speed to catch calm water)
  * oil detected      - otherwise

    python -m ml.quickcheck path/to/image.jpg
    python -m ml.quickcheck --calibrate Datasets/raw/sos/test/sentinel      (gate pass rate on real radar images)
"""
import argparse
import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from ml.config import DISAGREEMENT_FLAG, LOOKALIKE_CLASS, OIL_CLASS

MAX_SIDE = 1024            # images are shrunk to this size for speed
MAX_SATURATION = 0.12      # mean colour saturation above this -> a colour photo, not radar
MAX_SMOOTH_FRACTION = 0.35 # share of perfectly smooth pixels above this -> a photo, not speckled radar
SMOOTH_STD = 3.0           # local std (0-255 scale) below which a 7x7 patch counts as "perfectly smooth"
MIN_OIL_FRACTION = 0.005   # less than 0.5 % of the image -> no oil
LOW_CONFIDENCE = 0.6       # mean oil probability below this -> unsure


def load_image(data):
    """bytes / path / PIL image -> RGB PIL image (raises ValueError for unreadable files)."""
    try:
        img = data if isinstance(data, Image.Image) else Image.open(io.BytesIO(data) if isinstance(data, bytes) else data)
        img.load()
    except Exception as exc:
        raise ValueError("This file is not an image the system can read (use JPG or PNG; "
                         "iPhone HEIC photos must be converted first).") from exc
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def gate(img):
    """Is this a satellite radar image of the sea? Returns dict(passed, reasons, saturation, smooth_fraction)."""
    hsv = np.asarray(img.convert("HSV"), np.float32) / 255.0
    saturation = float(hsv[..., 1].mean())
    gray = np.asarray(img.convert("L"), np.float32)
    mean = ndimage.uniform_filter(gray, 7)
    std = np.sqrt(np.maximum(ndimage.uniform_filter(gray * gray, 7) - mean * mean, 0))
    usable = gray > 0                                   # radar scenes have black no-data borders: ignore them
    smooth = float(((std < SMOOTH_STD) & usable).sum() / max(usable.sum(), 1))
    reasons = []
    if saturation > MAX_SATURATION:
        reasons.append(f"it is a colour picture (colour level {saturation:.2f}); radar images are grey")
    if smooth > MAX_SMOOTH_FRACTION:
        reasons.append(f"{smooth * 100:.0f}% of it is perfectly smooth; radar images are grainy (speckle) everywhere")
    if usable.mean() < 0.2:
        reasons.append("the image is almost completely black")
    return {"passed": not reasons, "reasons": reasons,
            "saturation": round(saturation, 3), "smooth_fraction": round(smooth, 3)}


def _elongation(region):
    rows, cols = np.nonzero(region)
    if len(rows) < 5:
        return 1.0
    ev = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([cols, rows]))))
    return float(np.sqrt(ev[1] / max(ev[0], 1e-6)))


def overlay_png(gray, class_map, uncertainty=None):
    rgb = np.stack([gray] * 3, -1).astype(np.float32)
    oil = class_map == OIL_CLASS
    rgb[oil] = rgb[oil] * 0.35 + np.array([230, 40, 30]) * 0.65
    look = class_map == LOOKALIKE_CLASS
    rgb[look] = rgb[look] * 0.5 + np.array([250, 200, 0]) * 0.5
    if uncertainty is not None:
        u = np.clip(uncertainty / 0.4, 0, 1)[..., None] * 0.6
        rgb = rgb * (1 - u) + np.array([120, 40, 170]) * u
    buf = io.BytesIO()
    Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def check(data, detector=None, with_overlay=True):
    """Full quick check. Returns a JSON-ready dict."""
    img = load_image(data)
    g = gate(img)
    out = {"width": img.size[0], "height": img.size[1], "gate": g}
    if not g["passed"]:
        out.update(verdict="not_radar", title="Not a satellite radar image",
                   message="This picture cannot be analysed: " + "; ".join(g["reasons"]) + ". "
                           "The models were trained only on Sentinel-1 radar images of the sea.")
        return out

    if detector is None:
        from ml.segmentation.predict import SpillDetector
        detector = SpillDetector()
    gray = np.asarray(img.convert("L"), np.uint8)
    class_map, oil_prob = detector.predict(gray)                   # 0-valued pixels are treated as no data
    unc = getattr(detector, "last_uncertainty", None)
    oil = class_map == OIL_CLASS
    frac = float(oil.mean())
    conf = float(oil_prob[oil].mean()) if oil.any() else 0.0
    spread = float(unc[oil].mean()) if (unc is not None and oil.any()) else None
    labels, n = ndimage.label(oil)
    biggest = None
    if n:
        sizes = ndimage.sum(oil, labels, range(1, n + 1))
        biggest = labels == (int(np.argmax(sizes)) + 1)
        n = int((sizes >= 0.001 * oil.size).sum()) or 1          # count only patches of real size
    big_frac = float(biggest.mean()) if biggest is not None else 0.0
    elong = _elongation(biggest) if biggest is not None else None
    lookalike_px = float((class_map == LOOKALIKE_CLASS).mean())

    reasons = []
    if spread is not None and spread > DISAGREEMENT_FLAG:
        reasons.append(f"the models disagree (spread {spread:.2f})")
    if oil.any() and conf < LOW_CONFIDENCE:
        reasons.append(f"the model is unsure (mean oil probability {conf:.2f})")
    if lookalike_px > frac and lookalike_px > MIN_OIL_FRACTION:
        reasons.append("most dark areas have the round shape of look-alikes")

    if frac < MIN_OIL_FRACTION and not (lookalike_px > MIN_OIL_FRACTION):
        verdict, title = "no_oil", "No oil detected"
        message = f"Oil-like dark areas cover only {frac * 100:.2f}% of the image."
    elif reasons:
        verdict, title = "possible_lookalike", "Possibly a look-alike"
        message = (f"Dark areas cover {frac * 100:.1f}% of the image, but " + "; ".join(reasons) +
                   ". Look-alikes such as calm water, algae or rain cells also look dark in radar. Check the wind "
                   "speed at the image time or a second image before treating this as oil.")
    else:
        verdict, title = "oil", "Oil detected"
        message = (f"Oil covers {frac * 100:.1f}% of the image in {n} patch(es); mean model confidence "
                   f"{conf:.2f}" + (f", models agree (spread {spread:.2f})." if spread is not None else "."))
    out.update(verdict=verdict, title=title, message=message, detector=getattr(detector, "arch", ""),
               oil_fraction=round(frac, 4), confidence=round(conf, 3),
               model_disagreement=round(spread, 3) if spread is not None else None,
               patches=int(n), largest_patch_fraction=round(big_frac, 4),
               largest_patch_elongation=round(elong, 2) if elong else None)
    if with_overlay:
        out["overlay"] = overlay_png(gray, class_map, unc)
    return out


def evaluate(folder, limit=100, seed=0):
    """Image-level accuracy on labelled radar crops: does the verdict agree with the ground-truth mask?
    A crop counts as 'oil' if its mask has oil on at least MIN_OIL_FRACTION of the pixels."""
    from ml.segmentation.predict import SpillDetector
    folder = Path(folder)
    sats = sorted(folder.glob("*_sat.jpg"))
    rng = np.random.default_rng(seed)
    sats = [sats[i] for i in sorted(rng.choice(len(sats), min(limit, len(sats)), replace=False))]
    masks = {p.name: p for p in folder.parent.rglob("*_mask.png")}
    det = SpillDetector()
    rows = []
    for p in sats:
        m = masks.get(p.name.replace("_sat.jpg", "_mask.png"))
        if m is None:
            continue
        truth = float((np.asarray(Image.open(m).convert("L")) > 127).mean()) >= MIN_OIL_FRACTION
        v = check(p.read_bytes(), detector=det, with_overlay=False)["verdict"]
        rows.append((truth, v))
    if not rows:
        print("No masks found next to", folder)
        return
    tp = sum(t and v in ("oil", "possible_lookalike") for t, v in rows)
    tn = sum((not t) and v == "no_oil" for t, v in rows)
    fp = sum((not t) and v in ("oil", "possible_lookalike") for t, v in rows)
    fn = sum(t and v == "no_oil" for t, v in rows)
    flagged = sum(v == "possible_lookalike" for _, v in rows)
    rejected = sum(v == "not_radar" for _, v in rows)
    n = len(rows)
    print(f"{n} labelled radar crops ({sum(t for t, _ in rows)} with oil, {sum(not t for t, _ in rows)} without)")
    print(f"Accuracy (oil found vs not found): {(tp + tn) / n * 100:.1f}%")
    print(f"  oil crops found: {tp} of {tp + fn}   |  clean crops correctly 'no oil': {tn} of {tn + fp}")
    print(f"  flagged 'possibly a look-alike': {flagged}   |  wrongly rejected as not radar: {rejected}")


def calibrate(folder, limit=100):
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                   and "mask" not in p.stem)[:limit]
    res = [gate(load_image(p)) for p in files]
    ok = sum(r["passed"] for r in res)
    print(f"{ok} of {len(res)} radar images pass the gate ({ok / max(len(res), 1) * 100:.0f}%)")
    print(f"colour level max {max(r['saturation'] for r in res):.3f} (limit {MAX_SATURATION}); "
          f"smooth fraction max {max(r['smooth_fraction'] for r in res):.3f} (limit {MAX_SMOOTH_FRACTION})")
    for p, r in zip(files, res):
        if not r["passed"]:
            print("  rejected:", p.name, "-", "; ".join(r["reasons"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--calibrate", help="folder of radar images to test the gate on")
    ap.add_argument("--evaluate", help="folder of labelled radar crops (*_sat.jpg with *_mask.png nearby)")
    args = ap.parse_args()
    if args.evaluate:
        evaluate(args.evaluate)
    elif args.calibrate:
        calibrate(args.calibrate)
    elif args.image:
        r = check(Path(args.image).read_bytes(), with_overlay=False)
        print(f"{r['title']}: {r['message']}")
        print("gate:", r["gate"])
