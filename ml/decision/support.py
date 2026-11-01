"""Rule-based decision support: priority level, recommended response actions and alert text."""


def priority_level(spill, impact, top_suspect):
    score = impact["severity_score"]
    if spill.get("confidence") is not None and spill["confidence"] < 0.3:
        score *= 0.7          # weak detection -> verify first
    if spill.get("reliability") == "low":
        score *= 0.6          # calm sea / weak model output -> likely look-alike, verify first
    if top_suspect and top_suspect["score"] >= 0.6:
        score += 5
    if score >= 70:
        return "Critical", score
    if score >= 45:
        return "High", score
    if score >= 25:
        return "Medium", score
    return "Low", score


def recommend(spill, impact, suspects, forecast):
    top = suspects[0] if suspects else None
    level, score = priority_level(spill, impact, top)
    actions = []

    def add(action, detail, urgency):
        actions.append({"action": action, "detail": detail, "urgency": urgency})

    if spill.get("reliability") == "low" or (spill.get("confidence") is not None and spill["confidence"] < 0.5):
        reason = " ".join(spill.get("reliability_notes") or []) or "Detection confidence is low."
        add("Verify detection", f"{reason} Request the next Sentinel-1 pass or an aerial surveillance flight to confirm "
            "the slick is oil and not a look-alike.", "high")
    add("Notify authorities", "Send this incident report to the coast guard pollution response centre and "
        "the port state control office.", "high" if level in ("Critical", "High") else "medium")

    for site in impact["threatened_sites"][:5]:
        when = "oil is already within reach - deploy immediately" if site["eta_h"] < 1 else f"estimated in {site['eta_h']:.0f} h"
        add(f"Protect {site['name']}", f"Deploy containment booms before the oil arrives "
            f"({when}; site type: {site.get('type', 'sensitive area')}).",
            "high" if site["eta_h"] <= 24 else "medium")

    if forecast:
        f24 = next((f for f in forecast if f["hours"] == 24), forecast[-1])
        add("Dispatch recovery vessel", f"Send a skimmer vessel to the predicted slick centre at "
            f"{f24['centroid_lat']:.4f}, {f24['centroid_lon']:.4f} (24 h forecast).",
            "high" if level in ("Critical", "High") else "medium")

    if impact["first_landfall_h"] is not None:
        add("Shoreline response", f"Prepare beach clean-up teams; first landfall predicted after about "
            f"{impact['first_landfall_h']:.0f} h along ~{impact['coastline_affected_km']} km of coast.", "high")

    if top and top["score"] >= 0.35:
        add(f"Investigate {top['vessel_name']}", f"Inspect MMSI {top['mmsi']} at its next port call and collect "
            "oil samples for fingerprint comparison with the slick.", "high" if top["score"] >= 0.6 else "medium")
    elif top:
        add("No strong vessel match", f"The best match ({top['vessel_name']}, score {top['score']:.2f}) is weak. "
            "Consider non-vessel sources such as pipelines, offshore platforms or land run-off, and check for "
            "'dark' vessels with AIS switched off.", "high")
    else:
        add("Widen vessel search", "No AIS vessel matched. Check satellite AIS, radar and 'dark' vessels "
            "(transponder off) in the area.", "medium")

    for dv in (spill.get("nearby_dark_vessels") or [])[:3]:
        add("Check dark vessel", f"Radar shows a ship {dv['distance_km']:.1f} km from the spill at "
            f"{dv['lat']:.4f}, {dv['lon']:.4f} with no AIS signal. Send a patrol or aircraft to identify it "
            "(fixed structures such as oil platforms also look like bright targets).", "high")

    add("Sample the slick", "Collect oil samples from the slick for laboratory fingerprinting (evidence).", "medium")

    lines = [f"[{level.upper()}] Oil spill detected: {spill['area_km2']:.2f} sq km at "
             f"{spill['centroid_lat']:.4f}, {spill['centroid_lon']:.4f}."]
    if impact["threatened_sites"]:
        s = impact["threatened_sites"][0]
        lines.append(f"{s['name']} already threatened." if s["eta_h"] < 1 else f"{s['name']} threatened in about {s['eta_h']:.0f} h.")
    if top and top["score"] >= 0.35:
        lines.append(f"Top suspect: {top['vessel_name']} (MMSI {top['mmsi']}, score {top['score']:.2f}).")
    elif top:
        lines.append("No vessel strongly matches; a non-vessel source (e.g. pipeline) is possible.")
    if spill.get("nearby_dark_vessels"):
        lines.append(f"{len(spill['nearby_dark_vessels'])} ship(s) without AIS seen near the spill.")
    return {"priority": level, "priority_score": round(score, 1), "actions": actions, "alert_message": " ".join(lines)}
