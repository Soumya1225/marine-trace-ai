"""PDF evidence report for one incident (ReportLab + a Matplotlib situation map)."""
from datetime import datetime
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

from app.config import STORAGE_DIR

INK = colors.HexColor("#12324A")


def _situation_map(inc, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ml.impact.assessment import load_coastline, load_sites

    d = inc.data
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=130)
    lat, lon = load_coastline()
    if len(lat):
        ax.scatter(lon, lat, s=1, c="#6b7d6a", label="Coastline")
    for f in d.get("forecast") or []:
        if f["polygon"]:
            xs, ys = zip(*f["polygon"])
            ax.fill(xs, ys, alpha=0.12, color="#2C6E91")
            ax.plot(xs, ys, lw=0.6, color="#2C6E91")
            ax.annotate(f"+{f['hours']}h", (f["centroid_lon"], f["centroid_lat"]), fontsize=7, color="#2C6E91")
        if f.get("uncertainty_polygon"):
            xs, ys = zip(*f["uncertainty_polygon"])
            ax.plot(xs, ys, lw=0.5, ls=":", color="#2C6E91")
    poly = d["spill"].get("polygon") or []
    if poly:
        xs, ys = zip(*poly)
        ax.fill(xs, ys, color="#C8322B", alpha=0.8, label="Detected spill")
    if d.get("backtrack"):
        xs, ys = [p[0] for p in d["backtrack"]], [p[1] for p in d["backtrack"]]
        ax.plot(xs, ys, "--", color="#7A3B12", lw=1, label="Back-tracked drift")
    for s, c in zip(inc.suspects[:3], ["#C8322B", "#E8A42A", "#555555"]):
        if s.track:
            ax.plot([p["lon"] for p in s.track], [p["lat"] for p in s.track], color=c, lw=1.2,
                    label=f"#{s.rank} {s.vessel_name}")
    for site in load_sites():
        ax.plot(site["lon"], site["lat"], "^", color="#3E8E6A", ms=4)
    dark = [t for t in (d.get("ships") or {}).get("targets", []) if t.get("dark")]
    if dark:
        ax.scatter([t["lon"] for t in dark], [t["lat"] for t in dark], marker="x", c="#6B2FA0", s=25, label="Ship without AIS")
    infra = (d["spill"].get("infrastructure") or {}).get("nearby") or []
    plats = [f for f in infra if f["geometry"]["type"] == "Point"]
    for f in infra:
        if f["geometry"]["type"] == "LineString":
            xs, ys = zip(*f["geometry"]["coordinates"])
            ax.plot(xs, ys, "--", color="#F28C28", lw=1.6, label=f["name"])
    if plats:
        ax.scatter([f["geometry"]["coordinates"][0] for f in plats], [f["geometry"]["coordinates"][1] for f in plats],
                   marker="s", c="#F28C28", edgecolors="#111111", s=30, label="Oil platform", zorder=5)
    b = d.get("bounds")
    if b:
        ax.set_xlim(b[0] - 0.05, b[2] + 0.05)
        ax.set_ylim(b[1] - 0.05, b[3] + 0.05)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(fontsize=6, loc="lower left")
    ax.set_title("Situation map")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _overlay(inc, out_path):
    base = Path(inc.result_dir)
    sar = Image.open(base / "sar.png").convert("RGBA")
    mask = Image.open(base / "mask.png").convert("RGBA")
    sar.alpha_composite(mask)
    sar.thumbnail((1200, 1200))
    sar.convert("RGB").save(out_path)


SHAP_LABEL = {
    "min_dist_km": "Distance to the back-tracked spill", "alignment": "Course along the spill axis",
    "overlap": "Spill lies on its path", "hours_before": "Time before the image", "max_gap_h": "AIS transmission gap",
    "speed_ratio": "Slowed down near the spill", "loiter_h": "Loitering at low speed", "type_risk": "Vessel type",
    "length_m": "Vessel length",
}


def _shap_block(top, h2, small):
    """'Why this ship' table: SHAP contributions of the XGBoost ranker for the top suspect."""
    shap = ((top.features or {}).get("shap") if top is not None else None) or {}
    items = sorted(((k, v) for k, v in shap.items() if abs(v) >= 0.01), key=lambda kv: -abs(kv[1]))[:6]
    if not items:
        return []
    biggest = max(abs(v) for _, v in items)
    rows = [["Evidence", "Effect on the score", "Strength"]]
    for k, v in items:
        bar = "#" * max(1, round(abs(v) / biggest * 12))
        rows.append([SHAP_LABEL.get(k, k), "raises suspicion" if v > 0 else "lowers suspicion", bar])
    t = _table(rows, [7 * cm, 4 * cm, 5.5 * cm])
    return [KeepTogether([Paragraph(f"Why {top.vessel_name} is ranked #1 (SHAP explanation)", h2),
                          Paragraph("SHAP splits the XGBoost score into the contribution of each piece of evidence. "
                                    "Longer bars mean a bigger influence on this ship's score.", small),
                          Spacer(1, 4), t])]


def _table(rows, widths, header=True):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#B8C4CC"))]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), INK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    t.setStyle(TableStyle(style))
    return t


def build_report(inc, scene):
    out_dir = STORAGE_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"incident_{inc.id}.pdf"
    map_png, ov_png = out_dir / f"incident_{inc.id}_map.png", out_dir / f"incident_{inc.id}_sar.png"
    _situation_map(inc, map_png)
    _overlay(inc, ov_png)

    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], textColor=INK)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=INK)
    body = ss["BodyText"]
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10)
    d, spill, impact, dec = inc.data, inc.data["spill"], inc.data["impact"], inc.data["decision"]

    story = [Paragraph(f"Oil Spill Incident Report #{inc.id}", h1),
             Paragraph(f"Generated {datetime.utcnow():%Y-%m-%d %H:%M} UTC. Status: <b>{inc.status}</b>. "
                       f"Priority: <b>{inc.priority}</b> ({inc.priority_score}).", body), Spacer(1, 8)]
    story.append(_table([
        ["Satellite image", scene.filename],
        ["Acquisition time (UTC)", str(spill.get("acquisition_time"))],
        ["Location", f"{spill['centroid_lat']:.5f}, {spill['centroid_lon']:.5f}"],
        ["Area", f"{spill['area_km2']} sq km"],
        ["Length / width", f"{spill['length_km']} km / {spill['width_km']} km (elongation {spill['elongation']})"],
        ["Detection", f"{d.get('detector', {}).get('model', '')}, confidence {spill.get('confidence')}"],
        ["Wind at image time", f"{spill.get('wind_speed_ms', '-')} m/s, detection reliability: {spill.get('reliability', '-')}"],
        ["Estimated volume", f"{impact['volume']['m3'][0]}–{impact['volume']['m3'][1]} m³ "
                             f"({impact['volume']['barrels'][0]}–{impact['volume']['barrels'][1]} barrels)"],
    ], [5 * cm, 11.5 * cm], header=False))
    la = spill.get("lookalike_check")
    if la:
        rows = [["Evidence", "Value", "Points to", "Why"]]
        rows += [[e["name"], str(e["value"]), "oil" if e["effect"] > 0 else "look-alike", Paragraph(e["text"], small)]
                 for e in la["evidence"]]
        story.append(KeepTogether([
            Paragraph(f"Look-alike check: {la['verdict']} (oil score {la['oil_score']})", h2),
            Paragraph("Second stage after the deep models: physical evidence (wind, contrast, edge sharpness, shape, "
                      "texture) that separates oil from natural look-alikes such as calm water, algae and rain cells.", small),
            Spacer(1, 4), _table(rows, [3 * cm, 2.6 * cm, 2 * cm, 8.9 * cm])]))
    story += [Spacer(1, 8), Paragraph("Satellite image with detection", h2),
              RLImage(str(ov_png), width=16 * cm, height=12 * cm, kind="proportional"),
              PageBreak(), Paragraph("Situation map", h2),
              RLImage(str(map_png), width=16 * cm, height=12.8 * cm, kind="proportional")]

    story.append(Paragraph("Suspected vessels", h2))
    rows = [["Rank", "Vessel", "MMSI", "Type", "Score", "Closest (km)", "Review"]]
    for s in inc.suspects[:10]:
        rows.append([s.rank, s.vessel_name, s.mmsi, s.vessel_type, f"{s.score:.2f}",
                     s.features.get("min_dist_km"), s.review_status])
    story.append(_table(rows, [1.2 * cm, 4.5 * cm, 2.6 * cm, 2.6 * cm, 1.5 * cm, 2.2 * cm, 1.9 * cm]))
    for s in inc.suspects[:3]:
        story.append(Paragraph(f"<b>#{s.rank} {s.vessel_name}</b>: " + " ".join(s.evidence), small))
    story += _shap_block(inc.suspects[0] if inc.suspects else None, h2, small)

    if spill.get("reliability_notes"):
        story.append(Paragraph("Reliability note: " + " ".join(spill["reliability_notes"]), small))
    if spill.get("excluded_responders"):
        story.append(Paragraph("Excluded response vessels (present because of the incident, not suspects): "
                               + ", ".join(x["vessel_name"] for x in spill["excluded_responders"]), small))

    info = spill.get("infrastructure")
    if info:
        title = ("Possible non-ship source: platforms and pipelines" if info.get("possible_non_ship_source")
                 else "Oil platforms and pipelines nearby")
        parts = [Paragraph(title, h2), Paragraph(info["note"], body)]
        if info.get("nearby"):
            rows = [["Installation", "Type", "Operator", "From spill", "From drift path"]]
            rows += [[Paragraph(f["name"], small), f["kind"], Paragraph(f.get("operator", ""), small),
                      f"{f['distance_km']} km",
                      f"{f['drift_distance_km']} km" if f.get("drift_distance_km") is not None else "-"]
                     for f in info["nearby"]]
            parts += [Spacer(1, 4), _table(rows, [4.6 * cm, 2 * cm, 4.4 * cm, 2.6 * cm, 2.9 * cm])]
        parts.append(Paragraph("Installation positions are approximate (public NOAA / BOEM maps).", small))
        story.append(KeepTogether(parts))

    dvs = spill.get("nearby_dark_vessels") or []
    if dvs:
        rows = [["Position (lat, lon)", "Distance from spill", "Nearest AIS ship"]]
        rows += [[f"{v['lat']:.4f}, {v['lon']:.4f}", f"{v['distance_km']} km", f"{v.get('nearest_ais_km', '-')} km"] for v in dvs[:10]]
        block = _table(rows, [6 * cm, 4 * cm, 4 * cm])
    else:
        block = Paragraph("None detected within 15 km.", body)
    story.append(KeepTogether([Paragraph("Ships without AIS near the spill (dark vessels)", h2), block]))

    story.append(Paragraph("Drift forecast", h2))
    rows = [["After", "Centre (lat, lon)", "Beached"]]
    for f in d.get("forecast") or []:
        rows.append([f"{f['hours']} h", f"{f['centroid_lat']:.4f}, {f['centroid_lon']:.4f}", f"{f['beached_fraction'] * 100:.0f}%"])
    story.append(_table(rows, [3 * cm, 7 * cm, 3 * cm]))

    story.append(Paragraph("Impact", h2))
    landfall = (f"{impact['first_landfall_h']} h" if impact["first_landfall_h"] is not None else "not within 72 h")
    story.append(Paragraph(f"First landfall of the main slick: {landfall}. "
                           f"Coastline affected: {impact['coastline_affected_km']} km. Severity score: {impact['severity_score']}/100. "
                           "Sites marked 'already within reach' are close enough to the slick that oil can reach them at any time.", body))
    if impact["threatened_sites"]:
        rows = [["Site", "Type", "Oil reaches site", "Sensitivity"]]
        rows += [[t["name"], t.get("type", ""),
                  "already within reach" if (t.get("eta_h") or 0) <= 0 else f"in {t['eta_h']} h",
                  t["sensitivity"]] for t in impact["threatened_sites"]]
        story.append(_table(rows, [6.5 * cm, 4 * cm, 3 * cm, 3 * cm]))

    story.append(Paragraph("Recommended actions", h2))
    rows = [["Urgency", "Action", "Detail"]]
    rows += [[a["urgency"], Paragraph(a["action"], small), Paragraph(a["detail"], small)] for a in dec["actions"]]
    story.append(_table(rows, [2 * cm, 4 * cm, 10.5 * cm]))

    if inc.notes:
        story.append(Paragraph("Analyst notes", h2))
        for n in inc.notes:
            story.append(Paragraph(f"{n.created_at:%Y-%m-%d %H:%M} {n.user.name if n.user else ''}: {n.text}", small))

    ev = d.get("evidence")
    if ev:
        story.append(Paragraph("Evidence integrity", h2))
        story.append(Paragraph(f"Analysed {ev['analysed_at_utc']} with {ev['detector']}. SHA-256 fingerprints of the "
                               "input files: any later change to a file changes its fingerprint.", small))
        rows = [["Role", "File", "SHA-256"]] + [[f["role"], Paragraph(f["name"], small), Paragraph(f["sha256"], small)]
                                                 for f in ev["files"]]
        story.append(_table(rows, [2.6 * cm, 5.4 * cm, 8.5 * cm]))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Volume uses a 0.3–50 µm thickness range because SAR cannot measure thickness. "
                           "Vessel scores indicate where to investigate; they are not proof of liability.", small))
    SimpleDocTemplate(str(pdf), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      title=f"Incident {inc.id}").build(story)
    return pdf
