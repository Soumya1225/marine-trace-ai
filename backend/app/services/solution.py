"""Response solution for one incident: spill tier, best clean-up method, intercept plan, boom placement and a
time-phased response checklist.

Everything here is decision support built from the incident's own data (area, wind, forecast, threatened sites,
nearest response base). The rules follow general oil spill response practice (e.g. ITOPF technical guidance and the
tiered response idea of India's National Oil Spill Disaster Contingency Plan). They are planning suggestions for
an analyst, not operational orders: dispersant use and in-situ burning always need approval from the authority.
"""
import math

from app.services.response_plan import DEFAULTS, plan

OIL_DENSITY_T_PER_M3 = 0.9          # typical crude oil
TIER_LIMITS_T = (700, 10000)        # Tier 1 up to 700 t, Tier 2 up to 10,000 t, Tier 3 above (NOS-DCP)


def _km(lat1, lon1, lat2, lon2):
    r = math.pi / 180
    h = (math.sin((lat2 - lat1) * r / 2) ** 2 +
         math.cos(lat1 * r) * math.cos(lat2 * r) * math.sin((lon2 - lon1) * r / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(h))


def _in_india(lat, lon):
    return 5 <= lat <= 25 and 66 <= lon <= 95


# ------------------------------------------------------------------ tier
def tier(inc, impact, near_india):
    vol = (impact.get("volume") or {}).get("m3") or [0, 0]
    tonnes = [round(v * OIL_DENSITY_T_PER_M3, 1) for v in vol]
    base = 1 if tonnes[1] <= TIER_LIMITS_T[0] else 2 if tonnes[1] <= TIER_LIMITS_T[1] else 3
    level, reasons = base, [f"Estimated amount {tonnes[0]:g}–{tonnes[1]:g} tonnes (upper value used, to be on the safe side)."]
    sites = impact.get("threatened_sites") or []
    landfall = impact.get("first_landfall_h")
    if level < 3 and ((landfall is not None and landfall <= 24) or any(s.get("eta_h", 99) <= 24 for s in sites)):
        level += 1
        reasons.append("Raised one level: oil is expected to reach the coast or a sensitive site within 24 h.")
    if near_india:
        lead = {1: "Port / oil facility operator, supported by the Indian Coast Guard district",
                2: "Indian Coast Guard regional headquarters (Pollution Response Team)",
                3: "Indian Coast Guard Headquarters (national response, NOS-DCP)"}[level]
        inform = {1: ["Indian Coast Guard district", "Port authority", "State Pollution Control Board"],
                  2: ["Indian Coast Guard region / MRCC", "Port authority", "State Pollution Control Board",
                      "State Fisheries Department", "District Collector"],
                  3: ["Indian Coast Guard HQ", "Ministry of Environment, Forest and Climate Change",
                      "Directorate General of Shipping", "State governments of the affected coast", "Indian Navy"]}[level]
    else:
        lead = {1: "Port / facility operator with the local coast guard",
                2: "Coast guard regional command", 3: "National authority (coast guard headquarters)"}[level]
        inform = {1: ["Local coast guard", "Port authority"],
                  2: ["Coast guard region", "Port authority", "Environment agency", "Fisheries authority"],
                  3: ["National coast guard HQ", "Environment ministry", "Maritime administration", "Coastal state governments"]}[level]
    meaning = {1: "Small spill: local resources can handle it.",
               2: "Medium spill: needs regional resources and coordination.",
               3: "Major spill: national resources (and possibly international help) are needed."}[level]
    return {"tier": level, "tonnes": tonnes, "meaning": meaning, "lead_agency": lead, "inform": inform, "reasons": reasons}


# ------------------------------------------------------------------ clean-up method
def methods(spill, impact, first_eta_h):
    ws = spill.get("wind_speed_ms")
    ws = float(ws) if ws is not None else 5.0
    area = float(spill.get("area_km2") or 0)
    landfall = impact.get("first_landfall_h")
    sites = impact.get("threatened_sites") or []
    soon = min([s.get("eta_h", 99) for s in sites] + ([landfall] if landfall is not None else []) + [99])
    low_rel = spill.get("reliability") == "low" or (spill.get("confidence") or 1) < 0.5
    rows = []

    def add(key, name, what, score, plus, minus):
        rows.append({"key": key, "name": name, "what": what, "score": int(max(0, min(100, round(score)))),
                     "for": plus, "against": minus})

    # 1. booms + skimmers
    s, p, m = 60, [], []
    if ws <= 7:
        s += 25; p.append(f"Wind {ws:.1f} m/s: sea is calm enough for booms to hold the oil (booms fail above about 7–10 m/s).")
    elif ws <= 10:
        s += 5; m.append(f"Wind {ws:.1f} m/s: choppy sea, booms will lose some oil.")
    else:
        s -= 40; m.append(f"Wind {ws:.1f} m/s: waves wash oil over and under booms.")
    if first_eta_h is not None and (soon == 99 or first_eta_h < soon):
        s += 10; p.append(f"The nearest response vessel can arrive in about {first_eta_h:.0f} h, before the oil reaches the coast.")
    elif first_eta_h is not None:
        s -= 10; m.append(f"The nearest vessel needs about {first_eta_h:.0f} h; the oil may reach the coast first.")
    if area > 50:
        s -= 10; m.append(f"Large slick ({area:.1f} km²): many boom sections and vessels are needed.")
    p.append("Physically removes the oil from the sea; recovered oil can be stored and sampled as evidence.")
    add("mechanical", "Containment and recovery (booms + skimmers)",
        "Surround or sweep the slick with floating booms and pump it up with skimmers into storage tanks.", s, p, m)

    # 2. dispersants
    s, p, m = 45, [], []
    if 4 <= ws <= 12:
        s += 20; p.append(f"Wind {ws:.1f} m/s: enough wave energy to mix the dispersed oil into the water.")
    elif ws < 4:
        s -= 15; m.append(f"Wind {ws:.1f} m/s: too calm, dispersed oil would not mix down.")
    else:
        m.append(f"Wind {ws:.1f} m/s: spraying from aircraft or vessels becomes difficult.")
    if soon != 99 and soon <= 12:
        s -= 35; m.append("Oil is close to the coast or a sensitive site: dispersants are not used in shallow, near-shore water.")
    elif soon == 99 or soon > 24:
        s += 10; p.append("Oil stays offshore for at least 24 h: open water allows dispersant use.")
    if area > 20:
        s += 10; p.append("Large area: aircraft spraying can treat it faster than booms.")
    m.append("Needs approval from the coast guard / authority, and works best on fresh oil (first 1–2 days).")
    add("dispersant", "Chemical dispersants",
        "Spray dispersant so the oil breaks into tiny droplets that mix into the water and break down naturally.", s, p, m)

    # 3. in-situ burning
    s, p, m = 25, [], []
    if ws <= 5:
        s += 15; p.append(f"Wind {ws:.1f} m/s: calm enough to hold oil in a fire-resistant boom.")
    else:
        s -= 10; m.append(f"Wind {ws:.1f} m/s: too windy for safe burning.")
    if soon == 99 or soon > 48:
        s += 10; p.append("Far from the coast and people for at least 48 h.")
    else:
        s -= 20; m.append("Smoke and fire risk near the coast, ports or people.")
    m.append("Needs thick, fresh oil (several mm); radar cannot measure thickness and most slicks are thin.")
    m.append("Needs special approval and a fire boom; produces smoke.")
    add("burning", "In-situ burning",
        "Collect thick fresh oil in a fire-resistant boom and burn it on the sea surface.", s, p, m)

    # 4. shoreline protection
    s, p, m = 20, [], []
    if sites:
        s += 50; p.append(f"{len(sites)} sensitive site(s) are threatened, first in about {sites[0].get('eta_h', 0):.0f} h ({sites[0].get('name')}).")
    if landfall is not None:
        s += 25 if landfall <= 24 else 15
        p.append(f"Oil is forecast to reach the coast after about {landfall:.0f} h.")
    if not sites and landfall is None:
        m.append("The forecast does not show oil reaching the coast within 72 h.")
    p.append("Protecting beaches, mangroves and harbours is far cheaper than cleaning them afterwards.")
    add("shoreline", "Shoreline protection and clean-up",
        "Place protective booms at harbour entrances, mangroves and beaches before the oil arrives; prepare beach clean-up teams.", s, p, m)

    # 5. monitor only
    s, p, m = 20, [], []
    if area < 1 and not sites and landfall is None:
        s += 45; p.append(f"Small slick ({area:.2f} km²) staying offshore: likely to break up naturally.")
    if low_rel:
        s += 35; p.append("Detection reliability is low: confirm it is really oil before sending vessels.")
    if ws > 12:
        s += 25; p.append(f"Wind {ws:.1f} m/s: rough sea breaks the oil up naturally and makes other methods unsafe.")
    if sites or landfall is not None:
        s -= 20; m.append("Oil is heading for the coast: monitoring alone is not enough.")
    p.append("Keep tracking with the next satellite passes and aerial surveillance.")
    add("monitor", "Monitor and let it disperse naturally",
        "No active clean-up; track the slick with satellites and aircraft while waves and sunlight break it down.", s, p, m)

    if low_rel:                                   # unconfirmed detection: do not commit heavy resources yet
        for r in rows:
            if r["key"] in ("mechanical", "dispersant", "burning"):
                r["score"] = max(0, r["score"] - 20)
                r["against"].append("Detection not yet confirmed: verify before committing vessels or chemicals.")
    rows.sort(key=lambda r: -r["score"])
    best = rows[0]
    combo = [r for r in rows[1:] if r["score"] >= 60]
    summary = f"Recommended: {best['name']}."
    if combo:
        summary += " Combine with: " + ", ".join(r["name"].lower() for r in combo) + "."
    return {"recommended": best["key"], "summary": summary, "verify_first": low_rel, "wind_ms": ws, "methods": rows}


# ------------------------------------------------------------------ intercept + booms
def _track(inc, forecast):
    """Hourly slick centre from 0 to 72 h, interpolated between the forecast steps."""
    pts = [(0.0, inc.centroid_lat, inc.centroid_lon)] + [
        (float(f["hours"]), f["centroid_lat"], f["centroid_lon"]) for f in sorted(forecast or [], key=lambda f: f["hours"])]
    out = []
    for h in range(0, int(pts[-1][0]) + 1):
        for (h0, a0, o0), (h1, a1, o1) in zip(pts, pts[1:]):
            if h0 <= h <= h1:
                t = 0 if h1 == h0 else (h - h0) / (h1 - h0)
                out.append((h, a0 + t * (a1 - a0), o0 + t * (o1 - o0)))
                break
    return out or [(0, inc.centroid_lat, inc.centroid_lon)]


def intercept(inc, forecast, base, speed_kn, route_factor):
    speed_kmh = speed_kn * 1.852
    track = _track(inc, forecast)
    for h, la, lo in track:
        km = _km(base["lat"], base["lon"], la, lo) * route_factor
        if km / speed_kmh <= h or h == track[-1][0]:
            return {"base": base, "hours": h, "lat": round(la, 5), "lon": round(lo, 5), "distance_km": round(km, 1),
                    "travel_h": round(km / speed_kmh, 1), "reached": km / speed_kmh <= h,
                    "route": [[base["lat"], base["lon"]], [round(la, 5), round(lo, 5)]]}
    return None


def booms(impact, equipment):
    out = []
    for s in (impact.get("threatened_sites") or [])[:6]:
        eta = float(s.get("eta_h") or 0)
        out.append({"name": s.get("name"), "type": s.get("type", "sensitive site"), "lat": s["lat"], "lon": s["lon"],
                    "sensitivity": s.get("sensitivity"), "eta_h": eta, "deploy_by_h": round(max(eta - 2, 0), 1),
                    "boom_m": 300 if (s.get("sensitivity") or 0) >= 4 else 200,
                    "how": "Exclusion boom across the entrance" if "harbour" in str(s.get("type", "")).lower()
                    or "port" in str(s.get("type", "")).lower() else "Deflection boom to steer oil to a collection point"})
    return {"at_sea_boom_m": equipment.get("boom_m"), "sites": out}


# ------------------------------------------------------------------ checklist
def checklist(inc, spill, impact, t, m, icpt, bm, suspects):
    best = next(x for x in m["methods"] if x["key"] == m["recommended"])
    top = suspects[0] if suspects else None
    tasks = []

    def add(tid, phase, task, owner):
        tasks.append({"id": tid, "phase": phase, "task": task, "owner": owner})

    p1, p2, p3 = "0–6 h", "6–24 h", "24–72 h"
    if m["verify_first"]:
        add("verify", p1, "Confirm the slick is oil: aerial surveillance flight or the next satellite pass.", "Analyst / Coast Guard air unit")
    add("notify", p1, f"Notify {t['lead_agency']} (Tier {t['tier']}).", "Duty analyst")
    add("inform", p1, "Inform: " + ", ".join(t["inform"]) + ".", "Duty analyst")
    if icpt:
        add("mobilise", p1, f"Mobilise vessels and equipment from {icpt['base']['name']} (about {icpt['travel_h']:.0f} h away).", "Response base")
    add("warn", p1, "Warn fishermen and coastal communities to keep boats and nets away from the slick.", "District administration / Fisheries")
    add("evidence", p1, "Secure evidence: oil samples from the slick and AIS records" +
        (f" of {top['vessel_name']} (MMSI {top['mmsi']})." if top else "."), "Investigation officer")
    if icpt:
        add("intercept", p1 if icpt["hours"] <= 6 else p2 if icpt["hours"] <= 24 else p3, f"Reach the slick at the intercept point {icpt['lat']:.4f}, {icpt['lon']:.4f} (about {icpt['hours']} h after the image).", "Response vessel")
    add("method", p2, f"Start the recommended method: {best['name']}.", "On-scene commander")
    for i, s in enumerate(bm["sites"][:3]):
        when = "immediately (oil is already there)" if s["deploy_by_h"] <= 0 else f"before {s['deploy_by_h']:.0f} h"
        add(f"protect_{i}", p1 if s["deploy_by_h"] <= 6 else p2 if s["deploy_by_h"] <= 24 else p3,
            f"Protect {s['name']}: {s['how'].lower()} ({s['boom_m']} m), {when}.", "Shoreline team")
    add("monitor", p2, "Track the slick: aerial overflights and the next Sentinel-1 pass; update the forecast.", "Analyst")
    if impact.get("first_landfall_h") is not None:
        add("shore", p3, f"Shoreline survey and beach clean-up along about {impact.get('coastline_affected_km', 0)} km of coast.", "Shoreline team / local bodies")
    add("waste", p3, "Store and dispose of recovered oil and oily waste at an approved facility.", "Port / waste contractor")
    if top:
        add("investigate", p3, f"Inspect {top['vessel_name']} at its next port call and compare oil fingerprints with the slick samples.", "Port State Control / DG Shipping")
    add("report", p3, "Daily situation report to all agencies until the case is closed.", "Duty analyst")

    saved = (inc.data or {}).get("response_checklist") or {}
    for x in tasks:
        st = saved.get(x["id"]) or {}
        x["status"] = st.get("status", "pending")
        x["updated_by"], x["updated_at"] = st.get("by"), st.get("at")
    done = sum(1 for x in tasks if x["status"] == "done")
    return {"tasks": tasks, "done": done, "total": len(tasks), "progress": round(100 * done / max(len(tasks), 1))}


# ------------------------------------------------------------------ main
def build(inc, suspects=None):
    d = inc.data or {}
    spill, impact, forecast = d.get("spill") or {}, d.get("impact") or {}, d.get("forecast") or []
    rp = plan(inc)
    base = rp["nearest"][0] if rp["nearest"] else None
    near_india = _in_india(inc.centroid_lat, inc.centroid_lon) or (base and base["org"] == "Indian Coast Guard")
    t = tier(inc, impact, near_india)
    m = methods(spill, impact, base["eta_h"] if base else None)
    icpt = intercept(inc, forecast, base, DEFAULTS["vessel_speed_kn"], DEFAULTS["route_factor"]) if base else None
    bm = booms(impact, rp["equipment"])
    cl = checklist(inc, spill, impact, t, m, icpt, bm, suspects or [])
    return {
        "tier": t, "method": m, "intercept": icpt, "booms": bm, "checklist": cl,
        "slick": {"lat": inc.centroid_lat, "lon": inc.centroid_lon, "polygon": spill.get("polygon")},
        "forecast_path": [[p[1], p[2]] for p in _track(inc, forecast)],
        "note": ("Decision support only. Methods follow general oil spill response practice; dispersants and burning "
                 "need approval from the authority. Routes are straight-line guides; real routes follow safe shipping lanes."),
    }


TASK_STATUSES = ("pending", "in_progress", "done")
