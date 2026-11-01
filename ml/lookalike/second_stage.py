"""Second-stage look-alike check: is a detected dark patch really oil?

The segmentation models find dark, smooth patches. Several natural things also look dark in radar
(low-wind areas, rain cells, algae / biogenic films, current shear). Following the classic SAR oil-spill
literature, each detected patch is checked with physical evidence the deep models do not see directly:

  * wind speed     - below ~2 m/s the whole sea is smooth, so calm patches look like oil; above ~10-12 m/s
                     slicks break up and rain cells / wind shadows dominate
  * contrast       - oil damps the waves strongly: the patch is much darker than the water around it
  * edge sharpness - oil slicks have sharp borders; low-wind areas and rain cells fade out gradually
  * shape          - ship discharges are long and thin; huge round patches are typical of calm water
  * texture        - the inside of an oil slick is smooth and even
  * model agreement and confidence of the deep ensemble

Each piece of evidence raises or lowers an oil score (0-1) and gives a plain-language reason.
Scores >= 0.6 -> "Likely oil", 0.4-0.6 -> "Uncertain", < 0.4 -> "Likely look-alike".
The rule thresholds come from the literature. When models/lookalike_learned.pkl exists (scripts/train_lookalike_med.py,
trained on the expert-labelled Yang & Singha 2025 dataset), its oil probability is averaged with the rule score.
"""
import numpy as np
from scipy import ndimage

RING_PX = 10            # width of the background ring around the patch
OIL_AT, LOOK_AT = 0.6, 0.4
LEARNED_WEIGHT = 0.0     # share of the trained model in the oil score (0 = shown only; see check())


def _features(img, mask):
    """Contrast, edge sharpness and texture of one patch. img: uint8/float image, mask: bool (same crop)."""
    img = img.astype(np.float32)
    valid = img > 0                                           # 0 = no data
    ring = ndimage.binary_dilation(mask, iterations=RING_PX) & ~mask & valid
    inside = mask & valid
    if inside.sum() < 10 or ring.sum() < 10:
        return None
    m_in, m_out = float(img[inside].mean()), float(img[ring].mean())
    s_out = float(img[ring].std()) or 1.0
    smooth = ndimage.gaussian_filter(img, 1.5)
    grad = np.hypot(ndimage.sobel(smooth, 0), ndimage.sobel(smooth, 1))
    border = (mask ^ ndimage.binary_erosion(mask, iterations=2)) & valid
    g_border = float(grad[border].mean()) if border.any() else 0.0
    g_back = float(np.median(grad[ring])) or 1.0
    return {"contrast": round((m_out - m_in) / s_out, 2),          # darker than surroundings, in background std units
            "darkening": round(1 - m_in / max(m_out, 1e-6), 3),    # fraction darker
            "edge_sharpness": round(g_border / g_back, 2),         # border gradient vs background gradient
            "texture": round(float(img[inside].std()) / s_out, 2)}  # inside variability vs background


def check(img, mask, spill, wind_ms=None, use_learned=True, learned_result=None):
    """img/mask: crop around one detected patch. spill: dict with elongation, area_km2, confidence, ..."""
    f = _features(img, mask) or {}
    ev = []

    def add(name, value, delta, text):
        ev.append({"name": name, "value": value, "effect": round(delta, 2), "text": text})

    if wind_ms is not None:
        if wind_ms < 2:
            add("Wind", f"{wind_ms:.1f} m/s", -0.25, "Very low wind: the whole sea is smooth, so calm water looks like oil.")
        elif wind_ms < 3:
            add("Wind", f"{wind_ms:.1f} m/s", -0.1, "Low wind: calm patches are possible.")
        elif wind_ms <= 10:
            add("Wind", f"{wind_ms:.1f} m/s", 0.1, "Wind in the 3-10 m/s range where oil is seen best and look-alikes are rare.")
        elif wind_ms <= 12:
            add("Wind", f"{wind_ms:.1f} m/s", -0.05, "Strong wind: slicks start to break up and rain cells or wind shadows "
                                                     "become common, so dark patches need care.")
        else:
            add("Wind", f"{wind_ms:.1f} m/s", -0.15, "Very strong wind: oil is mixed into the water and rarely visible; "
                                                     "dark patches are usually rain cells or wind shadows.")
    c = f.get("contrast")
    if c is not None:
        if c >= 2:
            add("Contrast", c, 0.15, "Much darker than the water around it (strong wave damping, typical of oil).")
        elif c >= 1:
            add("Contrast", c, 0.05, "Moderately darker than the water around it.")
        else:
            add("Contrast", c, -0.15, "Only slightly darker than the surroundings (typical of natural films or calm water).")
    e = f.get("edge_sharpness")
    if e is not None:
        if e >= 1.8:
            add("Edge sharpness", e, 0.1, "Sharp borders, typical of an oil slick.")
        elif e < 1.2:
            add("Edge sharpness", e, -0.1, "Soft, fading borders, typical of low-wind areas and rain cells.")
    el, area = spill.get("elongation"), spill.get("area_km2")
    if el is not None:
        if el >= 3:
            add("Shape", f"elongation {el}", 0.1, "Long and thin, like a discharge from a moving ship.")
        elif area is not None and area > 20 and el < 1.6:
            add("Shape", f"elongation {el}, {area} km²", -0.15, "Very large and round, typical of a calm-water area.")
    t = f.get("texture")
    if t is not None:
        if t <= 0.8:
            add("Texture", t, 0.05, "Smooth and even inside, as oil is.")
        elif t > 1.3:
            add("Texture", t, -0.05, "Patchy inside (typical of algae films or rain cells).")
    conf, dis = spill.get("confidence"), spill.get("model_disagreement")
    if dis is not None and dis > 0.15:
        add("Model agreement", dis, -0.1, "The detection models disagree about this patch.")
    elif conf is not None and conf >= 0.7:
        add("Model confidence", conf, 0.05, "The detection models are confident.")

    score = float(np.clip(0.5 + sum(x["effect"] for x in ev), 0.02, 0.98))
    rule_score, lr = score, learned_result
    if lr is None and use_learned:
        from ml.lookalike import learned
        lr = learned.predict(img, mask)
    learned_p = lr["p"] if lr else None
    if lr:
        # trained on 2987 expert-labelled slicks and 6178 look-alike patches (Yang & Singha 2025, Mediterranean).
        # Outside the range of its training data it gets less say.
        # Display only: on the confirmed Huntington Beach spill (2021) the Mediterranean-trained model rated the real
        # slicks as look-alikes, so it is shown next to the rules but does not change the oil score.
        w = LEARNED_WEIGHT if len(lr["outside"]) <= 1 else LEARNED_WEIGHT * 0.4
        score = float(np.clip((1 - w) * rule_score + w * learned_p, 0.02, 0.98))
        text = ("Model trained on expert-labelled Mediterranean oil slicks and look-alikes rates this patch "
                + ("as oil-like." if learned_p >= 0.6 else "as look-alike-like." if learned_p < 0.4 else "as unclear."))
        if w == 0:
            text += " Shown for information only: it is not yet reliable outside its training region."
        elif w < LEARNED_WEIGHT:
            text += " This patch is unlike its training images (" + ", ".join(k.replace("_", " ") for k in lr["outside"]) \
                    + "), so it counts less."
        add("Trained look-alike model", f"{learned_p:.2f}", score - rule_score, text)
    verdict = "Likely oil" if score >= OIL_AT else "Likely look-alike" if score < LOOK_AT else "Uncertain"
    return {"oil_score": round(score, 2), "verdict": verdict, "features": f, "rule_score": round(rule_score, 2),
            "learned_oil_prob": None if learned_p is None else round(learned_p, 2),
            "learned_features": lr["features"] if lr else None, "learned_outside": lr["outside"] if lr else None,
            "evidence": sorted(ev, key=lambda x: -abs(x["effect"]))}


def pixel_of(transform, lat, lon):
    """lat/lon -> (row, col) for an affine transform (a, b, c, d, e, f)."""
    a, b, c, d, e, f = transform[:6]
    det = a * e - b * d
    col = (e * (lon - c) - b * (lat - f)) / det
    row = (-d * (lon - c) + a * (lat - f)) / det
    return row, col


def check_spills(img, oil_mask, transform, spills, wind_of=lambda sp: sp.get("wind_speed_ms")):
    """Run the check for every spill: matches each spill to the oil patch nearest its centre."""
    labels, n = ndimage.label(oil_mask)
    if n == 0:
        return
    objs = ndimage.find_objects(labels)
    centres = ndimage.center_of_mass(oil_mask, labels, range(1, n + 1))
    for sp in spills:
        r, c = pixel_of(transform, sp["centroid_lat"], sp["centroid_lon"])
        ri, ci = int(round(r)), int(round(c))
        if 0 <= ri < labels.shape[0] and 0 <= ci < labels.shape[1] and labels[ri, ci]:
            idx = int(labels[ri, ci])
        else:
            idx = 1 + int(np.argmin([(cy - r) ** 2 + (cx - c) ** 2 for cy, cx in centres]))
        rs, cs = objs[idx - 1]
        rs = slice(max(rs.start - RING_PX - 4, 0), min(rs.stop + RING_PX + 4, labels.shape[0]))
        cs = slice(max(cs.start - RING_PX - 4, 0), min(cs.stop + RING_PX + 4, labels.shape[1]))
        lr = None
        from ml.lookalike import learned
        if learned.load() is not None:
            # bigger crop so the background ring is still wide enough after resampling to the model's pixel size
            pm = learned.pixel_size_m(transform, sp["centroid_lat"])
            pad = int(np.ceil((learned.RING_PX + 4) * max(learned.TARGET_PIXEL_M / max(pm, 1.0), 1.0)))
            o_rs, o_cs = objs[idx - 1]
            br = slice(max(o_rs.start - pad, 0), min(o_rs.stop + pad, labels.shape[0]))
            bc = slice(max(o_cs.start - pad, 0), min(o_cs.stop + pad, labels.shape[1]))
            lr = learned.predict(img[br, bc], labels[br, bc] == idx, pixel_m=pm)
        res = check(img[rs, cs], labels[rs, cs] == idx, sp, wind_of(sp), use_learned=False, learned_result=lr)
        sp["lookalike_check"] = res
        if res["verdict"] == "Likely look-alike":
            notes = sp.setdefault("reliability_notes", [])
            msg = f"Second-stage look-alike check: oil score {res['oil_score']} ({res['evidence'][0]['text'].rstrip('.')})."
            notes[:] = [x for x in notes if not x.startswith("Second-stage look-alike check")] + [msg]
            sp["reliability"] = "low"
