"""Phase 2 of the response solution: follow-up satellite passes, coastal warning messages (English, Hindi, Odia),
clean-up progress log and a printable case file for the authorities."""
from datetime import datetime, timedelta
from html import escape

REPEAT_DAYS = 12          # each Sentinel-1 satellite repeats the same ground track every 12 days


# ------------------------------------------------------------------ 5. follow-up satellite passes
def _when(p):
    t = (p.get("ContentDate") or {}).get("Start") or ""
    return datetime.fromisoformat(t.replace("Z", "+00:00")).replace(tzinfo=None) if t else None


def followup(inc, search=None):
    """Sentinel-1 images of the spill area around the detection time, and the predicted next passes.

    Images taken before the detection show which satellite tracks cover this spot; each track comes back every
    12 days, so the next passes after the detection can be predicted. Images already in the catalogue after the
    detection are listed so the analyst can download and analyse them."""
    if search is None:
        from ml.preprocessing.download_sentinel1 import search
    t0 = inc.detected_at or datetime.utcnow()
    d = 0.05
    aoi = (inc.centroid_lon - d, inc.centroid_lat - d, inc.centroid_lon + d, inc.centroid_lat + d)
    start, end = t0 - timedelta(days=REPEAT_DAYS + 1), t0 + timedelta(days=REPEAT_DAYS + 1)
    products = search(aoi, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), cog_only=False)
    seen, images = set(), []
    for p in products:
        w = _when(p)
        if not w:
            continue
        key = (p["Name"][:3], w.strftime("%Y%m%d%H%M"))
        if key in seen:                                 # the COG and normal versions of the same image
            continue
        seen.add(key)
        images.append({"name": p["Name"], "satellite": p["Name"][:3], "time": w.isoformat() + "Z", "t": w})
    images.sort(key=lambda x: x["t"])

    tracks = {}
    for im in images:
        if im["t"] <= t0 + timedelta(minutes=5):        # tracks known up to the detection
            k = (im["satellite"], round((im["t"].hour * 60 + im["t"].minute) / 20))   # same satellite + time of day
            tracks[k] = im
    predicted = []
    for (sat, _), last in tracks.items():
        nxt = last["t"]
        while nxt <= t0 + timedelta(minutes=5):
            nxt += timedelta(days=REPEAT_DAYS)
        available = next((im for im in images if im["satellite"] == sat and abs((im["t"] - nxt).total_seconds()) < 3600), None)
        predicted.append({"satellite": sat, "time": nxt.isoformat() + "Z", "hours_after": round((nxt - t0).total_seconds() / 3600, 1),
                          "in_catalogue": available["name"] if available else None})
    predicted.sort(key=lambda x: x["time"])
    after = [{k: v for k, v in im.items() if k != "t"} | {"hours_after": round((im["t"] - t0).total_seconds() / 3600, 1)}
             for im in images if im["t"] > t0 + timedelta(minutes=5)]
    return {
        "detected_at": t0.isoformat() + "Z", "predicted": predicted, "images_after": after,
        "advice": ("Analyse the first image after the detection to confirm the slick and check the forecast. "
                   "Add a watch zone around the spill, then use Satellite feed → Download & analyse.")
        if after else ("No newer image in the catalogue yet. The table shows when the satellites will pass again; "
                       "add a watch zone with auto check so the next image is analysed automatically."),
        "note": "Each Sentinel-1 satellite repeats the same track every 12 days; passes predicted from the images before the detection.",
    }


# ------------------------------------------------------------------ 6. coastal warning messages
def warnings(inc, sol):
    d = inc.data or {}
    im = d.get("impact") or {}
    sites = im.get("threatened_sites") or []
    date = inc.detected_at.strftime("%d-%m-%Y %H:%M UTC") if inc.detected_at else "–"
    area, lat, lon = f"{inc.area_km2:.2f}", f"{inc.centroid_lat:.4f}", f"{inc.centroid_lon:.4f}"
    agency = sol["tier"]["lead_agency"].split(",")[0]
    india = "Indian Coast Guard" in sol["tier"]["lead_agency"]
    agency_short = "Indian Coast Guard" if india else "the Coast Guard"
    if sites:
        s = sites[0]
        site, eta = s["name"], s.get("eta_h") or 0
        when_en = "is already reaching" if eta <= 0 else f"may reach {site} in about {eta:.0f} hours"
        en_threat = f"The oil {when_en}{' ' + site if eta <= 0 else ''}."
        hi_threat = (f"तेल {site} तक पहुँच चुका है।" if eta <= 0 else f"अनुमान है कि तेल लगभग {eta:.0f} घंटे में {site} तक पहुँच सकता है।")
        or_threat = (f"ତେଲ {site} ପର୍ଯ୍ୟନ୍ତ ପହଞ୍ଚି ସାରିଛି।" if eta <= 0 else f"ଅନୁମାନ କରାଯାଉଛି ଯେ ତେଲ ପ୍ରାୟ {eta:.0f} ଘଣ୍ଟା ମଧ୍ୟରେ {site} ପର୍ଯ୍ୟନ୍ତ ପହଞ୍ଚିପାରେ।")
    else:
        en_threat = "No coast is directly threatened at the moment, but stay alert."
        hi_threat = "अभी तट को कोई सीधा खतरा नहीं है, लेकिन सावधान रहें।"
        or_threat = "ବର୍ତ୍ତମାନ ଉପକୂଳକୁ ସିଧାସଳଖ ବିପଦ ନାହିଁ, କିନ୍ତୁ ସତର୍କ ରୁହନ୍ତୁ।"
    en = "\n".join([
        "⚠ OIL SPILL WARNING",
        f"An oil spill of about {area} sq km was seen by satellite on {date} at {lat}, {lon}.",
        en_threat,
        "Fishermen: do not take boats or nets into this area.",
        "Coastal residents: do not touch the oil; keep children and animals away from the beach.",
        f"If you see oil, dead fish or oiled birds on the shore, inform {agency_short} immediately.",
        f"— {agency}"])
    hi = "\n".join([
        "⚠ तेल रिसाव चेतावनी",
        f"{date} को उपग्रह द्वारा समुद्र में लगभग {area} वर्ग किमी का तेल रिसाव देखा गया है, स्थान: {lat}, {lon}।",
        hi_threat,
        "मछुआरे कृपया इस क्षेत्र में नाव और जाल न ले जाएँ।",
        "तटीय निवासी तेल को न छुएँ, बच्चों और पशुओं को समुद्र तट से दूर रखें।",
        f"तट पर तेल, मरी हुई मछलियाँ या तेल से सने पक्षी दिखें तो तुरंत {'भारतीय तटरक्षक बल' if india else 'तटरक्षक बल'} को सूचित करें।",
        f"— {agency}"])
    od = "\n".join([
        "⚠ ତେଲ ଲିକ୍ ଚେତାବନୀ",
        f"{date} ରେ ଉପଗ୍ରହ ଦ୍ୱାରା ସମୁଦ୍ରରେ ପ୍ରାୟ {area} ବର୍ଗ କିମି ତେଲ ଲିକ୍ ଦେଖାଯାଇଛି, ସ୍ଥାନ: {lat}, {lon}।",
        or_threat,
        "ମତ୍ସ୍ୟଜୀବୀମାନେ ଦୟାକରି ଏହି ଅଞ୍ଚଳକୁ ଡଙ୍ଗା ଓ ଜାଲ ନେଇ ଯାଆନ୍ତୁ ନାହିଁ।",
        "ଉପକୂଳବାସୀ ତେଲକୁ ଛୁଅନ୍ତୁ ନାହିଁ, ପିଲା ଓ ପଶୁମାନଙ୍କୁ ସମୁଦ୍ର କୂଳରୁ ଦୂରେଇ ରଖନ୍ତୁ।",
        f"କୂଳରେ ତେଲ, ମୃତ ମାଛ କିମ୍ବା ତେଲ ଲାଗିଥିବା ପକ୍ଷୀ ଦେଖିଲେ ତୁରନ୍ତ {'ଭାରତୀୟ ତଟରକ୍ଷୀ ବାହିନୀ' if india else 'ତଟରକ୍ଷୀ ବାହିନୀ'}କୁ ଜଣାନ୍ତୁ।",
        f"— {agency}"])
    return {"en": en, "hi": hi, "or": od,
            "note": "Hindi and Odia text is prepared automatically from templates; have a native speaker check it before sending. "
                    "Place names are kept in English."}


# ------------------------------------------------------------------ 7. clean-up progress
def cleanup(inc, rp):
    log_rows = list((inc.data or {}).get("cleanup_log") or [])
    oil = round(sum(r.get("oil_m3") or 0 for r in log_rows), 2)
    shore = round(sum(r.get("shore_km") or 0 for r in log_rows), 2)
    cost = round(sum(r.get("cost") or 0 for r in log_rows))
    t_oil = (rp["equipment"]["recoverable_m3"] or [0, 0])[1]
    t_shore = rp["cost"]["shoreline_km"] or 0
    t_cost = (rp["cost"]["total"] or [0, 0])[1]
    pct = lambda a, b: min(100, round(100 * a / b)) if b else None
    return {"entries": [{**r, "index": i} for i, r in enumerate(log_rows)],
            "totals": {"oil_m3": oil, "shore_km": shore, "cost": cost},
            "targets": {"oil_m3": t_oil, "shore_km": t_shore, "cost": t_cost, "currency": rp["cost"]["currency"]},
            "progress": {"oil": pct(oil, t_oil), "shore": pct(shore, t_shore), "budget": pct(cost, t_cost)},
            "note": "Targets come from the Response estimate: recoverable oil (upper value), affected coastline and total cost (upper value)."}


# ------------------------------------------------------------------ 8. case file for the authorities
def _fmt_money(v, cur):
    return f"{cur} {int(v or 0):,}"


def casefile_html(inc, scene, suspects, sol, rp, user_name=""):
    d = inc.data or {}
    sp, im, ev = d.get("spill") or {}, d.get("impact") or {}, d.get("evidence") or {}
    t, cur = sol["tier"], rp["cost"]["currency"]
    top = suspects[0] if suspects else None
    rows = lambda pairs: "".join(f"<tr><th>{escape(str(k))}</th><td>{v}</td></tr>" for k, v in pairs)
    sus_rows = "".join(
        f"<tr><td>{s.rank}</td><td><b>{escape(s.vessel_name)}</b><br><span class=m>MMSI {s.mmsi}, {escape(s.vessel_type or '')}</span></td>"
        f"<td>{s.score:.2f}</td><td>{escape(s.review_status or 'pending')}</td>"
        f"<td><ul>{''.join(f'<li>{escape(str(e))}</li>' for e in (s.evidence or [])[:4])}</ul></td></tr>"
        for s in suspects[:5]) or "<tr><td colspan=5>No vessel matched in the AIS data.</td></tr>"
    files = "".join(f"<tr><td>{escape(f.get('role', ''))}</td><td>{escape(f.get('name', ''))}</td><td class=mono>{escape(f.get('sha256', ''))}</td></tr>"
                    for f in ev.get("files") or []) or "<tr><td colspan=3>–</td></tr>"
    vol = (im.get("volume") or {}).get("m3") or [0, 0]
    total = rp["cost"]["total"]
    case_no = f"SW-{(inc.detected_at or datetime.utcnow()):%Y}-{inc.id:04d}"
    status_txt = {"confirmed": "Confirmed oil spill", "rejected": "Rejected (false alarm)", "closed": "Closed"}.get(inc.status, "Under investigation")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Case file {case_no}</title>
<style>body{{font-family:Arial,sans-serif;color:#1b2a33;max-width:900px;margin:24px auto;padding:0 16px;font-size:14px}}
h1{{font-size:22px;margin:0}}h2{{font-size:16px;border-bottom:2px solid #0F2A3D;padding-bottom:3px;margin-top:22px}}
table{{border-collapse:collapse;width:100%;margin:6px 0}}th,td{{border:1px solid #ccd5da;padding:5px 7px;text-align:left;vertical-align:top}}
th{{background:#F1F4F5;width:30%}}.m{{color:#5b6b75;font-size:12px}}.mono{{font-family:monospace;font-size:11px;word-break:break-all}}
.box{{border:1px solid #ccd5da;padding:8px 10px;background:#FAFBFB}}ul{{margin:0;padding-left:16px}}
.top{{display:flex;justify-content:space-between;align-items:flex-start}}@media print{{button{{display:none}}}}</style></head><body>
<div class=top><div><h1>Oil pollution case file</h1><div class=m>Case {case_no} · Incident #{inc.id} · prepared {datetime.utcnow():%d %b %Y %H:%M} UTC{(' by ' + escape(user_name)) if user_name else ''}</div></div>
<button onclick="window.print()">Print / Save as PDF</button></div>

<h2>1. Incident</h2><table>{rows([
        ("Status", status_txt), ("Location", f"{inc.centroid_lat:.5f}, {inc.centroid_lon:.5f}"),
        ("Satellite image time", f"{inc.detected_at:%d %b %Y %H:%M} UTC" if inc.detected_at else "–"),
        ("Satellite image", escape(scene.filename if scene else "–")),
        ("Slick area", f"{inc.area_km2:.3f} km²"), ("Estimated amount", f"{vol[0]:g}–{vol[1]:g} m³ (radar cannot measure thickness; wide range)"),
        ("Detection confidence", f"{inc.confidence:.2f}" if inc.confidence is not None else "–"),
        ("Wind at detection", f"{sp.get('wind_speed_ms', '–')} m/s"), ("Priority / tier", f"{inc.priority} / Tier {t['tier']}")])}</table>

<h2>2. Suspected source</h2>
{f'<p class=box>Main suspect: <b>{escape(top.vessel_name)}</b> (MMSI {top.mmsi}), score {top.score:.2f}, review status: {escape(top.review_status or "pending")}. '
 'The ranking compares each ship&#39;s AIS track with the oil drifted back in time (distance, timing, course along the slick, slow-downs, AIS gaps).</p>' if top else
 '<p class=box>No AIS vessel matched. Consider ships with AIS switched off, pipelines and platforms.</p>'}
<table><tr><th style="width:5%">#</th><th style="width:25%">Vessel</th><th style="width:8%">Score</th><th style="width:10%">Review</th><th>Evidence</th></tr>{sus_rows}</table>

<h2>3. Digital evidence</h2>
<p class=m>SHA-256 fingerprints of the input files (analysed {escape(str(ev.get('analysed_at_utc', '–')))}). A changed file gives a different fingerprint.</p>
<table><tr><th>Role</th><th>File</th><th>SHA-256</th></tr>{files}</table>

<h2>4. Samples and inspection requested</h2><ul>
<li>Collect at least two oil samples from the slick (sealed, labelled with time and position, chain-of-custody form).</li>
{f'<li>Collect oil samples from {escape(top.vessel_name)} (bilge, tanks, slop tanks) at its next port call for fingerprint comparison.</li><li>Check the Oil Record Book and the oily-water separator of {escape(top.vessel_name)}.</li>' if top else ''}
<li>Keep the AIS records and satellite images listed above unchanged.</li></ul>

<h2>5. Damage estimate (polluter pays)</h2><table>{rows([
        ("Clean-up at sea", f"{_fmt_money(rp['cost']['at_sea'][0], cur)} – {_fmt_money(rp['cost']['at_sea'][1], cur)}"),
        ("Shoreline clean-up", f"{_fmt_money(rp['cost']['shoreline'], cur)} ({rp['cost']['shoreline_km']} km of coast)"),
        ("Total estimate", f"<b>{_fmt_money(total[0], cur)} – {_fmt_money(total[1], cur)}</b>"),
        ("Threatened sites", escape(", ".join(s['name'] for s in (im.get('threatened_sites') or [])[:6]) or "none within 72 h"))])}</table>
<p class=m>Estimate from the response plan with example unit costs; replace with actual expenses when claiming costs from the polluter.</p>

<h2>6. Response</h2><table>{rows([
        ("Lead agency", escape(t['lead_agency'])), ("Recommended method", escape(sol['method']['summary'])),
        ("Response checklist", f"{sol['checklist']['done']} of {sol['checklist']['total']} tasks done")])}</table>

<p class=m style="margin-top:24px">Prepared automatically by SpillWatch (Marine Trace AI) as decision support. Findings must be verified by the
investigating officer before legal action.</p></body></html>"""

