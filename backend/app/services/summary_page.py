"""One-page incident summary as HTML, in English or Hindi.

The browser renders Hindi (Devanagari) correctly, and "Print -> Save as PDF" turns it into a PDF.
(ReportLab cannot shape Devanagari conjuncts properly, so the Hindi version is HTML, not ReportLab.)
"""
from html import escape

from app.services.response_plan import plan

T = {
    "en": {
        "title": "Oil spill incident summary", "incident": "Incident", "priority": "Priority", "status": "Status",
        "image_time": "Satellite image time (UTC)", "location": "Location (lat, lon)", "area": "Spill area",
        "confidence": "Detection confidence", "wind": "Wind", "reliability": "Reliability",
        "lookalike": "Look-alike check", "volume": "Estimated volume", "landfall": "First landfall",
        "not72": "not within 72 hours", "after": "after about {h} hours", "sites": "Threatened sensitive sites",
        "none": "None within 72 hours", "suspects": "Suspect vessels (where to investigate)", "rank": "Rank",
        "vessel": "Vessel", "type": "Type", "score": "Score", "actions": "Recommended actions",
        "response": "Nearest response base", "eta": "travel time about {h} h",
        "boom": "Containment boom needed: about {m} m", "cost": "Cost estimate", "zones": "Watch zone",
        "disclaimer": "Vessel scores show where to investigate; they are not proof of liability. Volume is a wide range "
                      "because radar cannot measure oil thickness. Costs use example unit rates.",
        "print": "Print / Save as PDF", "status_map": {"detected": "Detected", "under_review": "Under review",
                                                        "confirmed": "Confirmed", "closed": "Closed", "rejected": "Rejected"},
        "pri_map": {}, "normal": "normal", "low": "low",
    },
    "hi": {
        "title": "तेल रिसाव घटना सारांश", "incident": "घटना", "priority": "प्राथमिकता", "status": "स्थिति",
        "image_time": "उपग्रह चित्र का समय (UTC)", "location": "स्थान (अक्षांश, देशांतर)", "area": "रिसाव का क्षेत्रफल",
        "confidence": "पहचान का विश्वास स्तर", "wind": "हवा की गति", "reliability": "विश्वसनीयता",
        "lookalike": "समान दिखने वाली वस्तु की जाँच", "volume": "अनुमानित मात्रा", "landfall": "तट तक पहुँचने का समय",
        "not72": "72 घंटों के भीतर नहीं", "after": "लगभग {h} घंटे बाद", "sites": "खतरे में संवेदनशील स्थान",
        "none": "72 घंटों के भीतर कोई नहीं", "suspects": "संदिग्ध जहाज़ (जाँच के लिए)", "rank": "क्रम",
        "vessel": "जहाज़", "type": "प्रकार", "score": "अंक", "actions": "अनुशंसित कार्रवाई",
        "response": "निकटतम प्रतिक्रिया केंद्र", "eta": "पहुँचने में लगभग {h} घंटे",
        "boom": "आवश्यक रोकथाम बूम: लगभग {m} मीटर", "cost": "अनुमानित लागत", "zones": "निगरानी क्षेत्र",
        "disclaimer": "जहाज़ों के अंक केवल यह बताते हैं कि जाँच कहाँ करनी है; ये दोष का प्रमाण नहीं हैं। "
                      "रडार तेल की मोटाई नहीं माप सकता, इसलिए मात्रा एक व्यापक सीमा में दी गई है। "
                      "लागत उदाहरण दरों पर आधारित है।",
        "print": "प्रिंट करें / PDF के रूप में सहेजें",
        "status_map": {"detected": "पहचाना गया", "under_review": "जाँच जारी", "confirmed": "पुष्टि हुई",
                       "closed": "बंद", "rejected": "गलत चेतावनी"},
        "pri_map": {"Critical": "अति गंभीर", "High": "उच्च", "Medium": "मध्यम", "Low": "निम्न"},
        "verdict_map": {"Likely oil": "संभवतः तेल", "Uncertain": "अनिश्चित", "Likely look-alike": "संभवतः तेल जैसा दिखने वाला प्राकृतिक दृश्य"},
        "site_map": {"public beach": "सार्वजनिक समुद्र तट", "beach": "समुद्र तट", "wetland": "आर्द्रभूमि", "port": "बंदरगाह",
                     "marine protected area": "समुद्री संरक्षित क्षेत्र", "mangrove": "मैंग्रोव वन", "coral reef": "प्रवाल भित्ति",
                     "fishing area": "मत्स्य क्षेत्र", "nature reserve": "प्रकृति अभयारण्य"},
        "normal": "सामान्य", "low": "कम",
    },
}


def _row(k, v):
    return f"<tr><th>{escape(str(k))}</th><td>{v}</td></tr>"


def build(inc, lang="en"):
    t = T.get(lang, T["en"])
    d = inc.data or {}
    sp, im, dec = d.get("spill") or {}, d.get("impact") or {}, d.get("decision") or {}
    pri = t["pri_map"].get(inc.priority, inc.priority)
    rows = [
        _row(t["priority"], f"<b>{escape(pri)}</b>"),
        _row(t["status"], escape(t["status_map"].get(inc.status, inc.status))),
        _row(t["image_time"], escape(inc.detected_at.strftime("%Y-%m-%d %H:%M")) if inc.detected_at else "–"),
        _row(t["location"], f"{inc.centroid_lat:.4f}, {inc.centroid_lon:.4f}"),
        _row(t["area"], f"{inc.area_km2:.2f} km²"),
    ]
    if d.get("zones"):
        rows.append(_row(t["zones"], escape(", ".join(d["zones"]))))
    if inc.confidence is not None:
        rows.append(_row(t["confidence"], f"{inc.confidence:.2f}"))
    if sp.get("wind_speed_ms") is not None:
        rows.append(_row(t["wind"], f"{sp['wind_speed_ms']} m/s"))
    rel = sp.get("reliability", "normal")
    rows.append(_row(t["reliability"], escape(t.get(rel, rel))))
    la = sp.get("lookalike_check")
    if la:
        verdict = t.get("verdict_map", {}).get(la["verdict"], la["verdict"])
        rows.append(_row(t["lookalike"], f"{escape(verdict)} ({la['oil_score']})"))
    vol = (im.get("volume") or {}).get("m3")
    if vol:
        rows.append(_row(t["volume"], f"{vol[0]:.1f}–{vol[1]:.0f} m³"))
    land = im.get("first_landfall_h")
    rows.append(_row(t["landfall"], t["after"].format(h=land) if land is not None else t["not72"]))

    sites = im.get("threatened_sites") or []
    smap = t.get("site_map", {})
    sites_html = ("<ul>" + "".join(f"<li>{escape(s['name'])} ({escape(smap.get(s['type'], s['type']))})</li>" for s in sites[:8]) + "</ul>"
                  if sites else f"<p>{t['none']}</p>")
    sus = "".join(f"<tr><td>{s.rank}</td><td>{escape(s.vessel_name)} <small>MMSI {s.mmsi}</small></td>"
                  f"<td>{escape(s.vessel_type)}</td><td>{s.score:.2f}</td></tr>" for s in list(inc.suspects)[:5])
    acts_note = "" if lang == "en" else "<p><small>(कार्रवाई का विवरण अंग्रेज़ी में)</small></p>"
    acts = "".join(f"<li><b>{escape(a['action'])}</b>" + (f"<br><small>{escape(a.get('detail', ''))}</small>" if a.get("detail") else "")
                   + "</li>" for a in (dec.get("actions") or [])[:6])
    rp = plan(inc)
    near = rp["nearest"][0] if rp["nearest"] else None
    c = rp["cost"]
    resp = ""
    if near:
        boom = t["boom"].format(m=f"{rp['equipment']['boom_m']:,}")
        resp = (f"<p><b>{escape(near['name'])}</b>: {near['distance_km']:.0f} km, {t['eta'].format(h=near['eta_h'])}</p>"
                f"<p>{boom}</p>"
                f"<p>{t['cost']}: {c['currency']} {c['total'][0]:,} – {c['total'][1]:,}</p>")

    return f"""<!doctype html>
<html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(t['title'])} #{inc.id}</title>
<style>
 body {{ font-family: 'Nirmala UI', 'Noto Sans Devanagari', 'Mangal', 'Segoe UI', Arial, sans-serif; color: #0F2A3D; margin: 0; background: #EEF2F0; }}
 main {{ max-width: 820px; margin: 1.5rem auto; background: #fff; padding: 2rem 2.2rem; border-top: 6px solid #0F2A3D; }}
 h1 {{ margin: 0 0 .2rem; font-size: 1.6rem; }} h2 {{ font-size: 1.15rem; margin: 1.4rem 0 .5rem; border-bottom: 1px solid #C9D3D1; padding-bottom: .2rem; }}
 table {{ border-collapse: collapse; width: 100%; }} th, td {{ text-align: left; padding: .35rem .5rem; border-bottom: 1px solid #E3EAE8; vertical-align: top; }}
 th {{ width: 40%; color: #4A6070; font-weight: 600; }} .sus th {{ width: auto; }}
 small {{ color: #4A6070; }} .note {{ margin-top: 1.4rem; font-size: .85rem; color: #4A6070; }}
 .bar {{ max-width: 820px; margin: 1rem auto 0; text-align: right; }} button {{ font: inherit; padding: .45rem .9rem; cursor: pointer; }}
 @media print {{ .bar {{ display: none; }} body {{ background: #fff; }} main {{ margin: 0; }} }}
</style></head>
<body>
<div class="bar"><button onclick="window.print()">{escape(t['print'])}</button></div>
<main>
 <h1>{escape(t['title'])}</h1>
 <p><small>{escape(t['incident'])} #{inc.id} · {escape(inc.title)}</small></p>
 <table>{''.join(rows)}</table>
 <h2>{escape(t['sites'])}</h2>{sites_html}
 <h2>{escape(t['suspects'])}</h2>
 <table class="sus"><tr><th>{t['rank']}</th><th>{t['vessel']}</th><th>{t['type']}</th><th>{t['score']}</th></tr>{sus}</table>
 <h2>{escape(t['actions'])}</h2>{acts_note}<ol>{acts}</ol>
 <h2>{escape(t['response'])}</h2>{resp}
 <p class="note">{escape(t['disclaimer'])}</p>
</main></body></html>"""
