"""Incident assistant: answers questions about one incident from its own data.

Works offline with a rule-based engine (intent matching over the incident's spill, suspects, SHAP values, forecast,
impact, decision and case data). If ANTHROPIC_API_KEY is set in .env, the same facts are given to a Claude model so
answers are phrased more naturally; the model is told to use only those facts.
Every answer lists which part of the incident data it came from, so an analyst can check it.
"""
import json
import os
import re
import urllib.request

from app.services.serializers import case_fields

SHAP_LABEL = {
    "min_dist_km": "distance to the back-tracked spill", "alignment": "course along the spill axis",
    "overlap": "spill lying on its path", "hours_before": "time before the image", "max_gap_h": "AIS gap",
    "speed_ratio": "slowing down near the spill", "loiter_h": "loitering", "type_risk": "vessel type",
    "length_m": "vessel length",
}

INTENTS = [
    ("dark", r"dark|no ais|without ais|ais off|switched off|silent|platform|fixed structure"),
    ("why", r"\bwhy\b|reason|explain|shap|evidence against"),
    ("forecast", r"forecast|where .*\b(be|go|move|drift|spread)|\bin \d+\s*(h|hour)|next \d+|tomorrow|spread|drift"),
    ("impact", r"beach|coast|landfall|site|threat|impact|damage|reach|sensitive"),
    ("reliability", r"look.?alike|really|real oil|oil or|\bsure\b|confiden|reliab|wind|false|calm|trust|uncertain"),
    ("actions", r"action|what should|should we|to do\b|recommend|priority|urgent|respond|response|next step"),
    ("case", r"assign|status|deadline|due|overdue|\bcase\b|who is (working|handling)"),
    ("suspects", r"suspect|ship|vessel|who (did|caused|is responsible)|culprit|polluter|responsible|rank"),
    ("size", r"size|area|big|large|volume|barrel|how much oil|severity"),
    ("detection", r"detect|model|u-?net|segformer|deeplab|ensemble|how (was|did)|method"),
    ("source", r"pipeline|non.?ship|infrastructure|source|origin|come from|back.?track"),
]

SUGGESTIONS = ["Which ship is most suspicious and why?", "Where will the oil be in 24 hours?",
               "Which beaches are threatened?", "Is this really oil or a look-alike?",
               "What should we do now?", "Were there any ships without AIS?"]


def _intent(q):
    q = q.lower()
    hits = [name for name, rx in INTENTS if re.search(rx, q)]
    if "why" in hits and ("suspects" in hits or re.search(r"suspicious|ship|vessel", q)):
        return "why"
    return hits[0] if hits else "summary"


def _why(s):
    f = s.features or {}
    shap = f.get("shap") or {}
    up = sorted([(k, v) for k, v in shap.items() if v > 0.005], key=lambda x: -x[1])[:3]
    down = sorted([(k, v) for k, v in shap.items() if v < -0.005], key=lambda x: x[1])[:2]
    parts = [f"{s.vessel_name} (MMSI {s.mmsi}, {s.vessel_type}) is ranked #{s.rank} with score {s.score:.2f} "
             f"(rule score {s.rule_score:.2f}" + (f", XGBoost {s.ml_score:.2f}" if s.ml_score is not None else "") + ")."]
    if s.evidence:
        parts.append("Evidence: " + " ".join(s.evidence))
    if up:
        parts.append("The model scored it higher because of " + ", ".join(str(SHAP_LABEL.get(k, k)) for k, _ in up) + ".")
    if down:
        parts.append("It scored lower because of " + ", ".join(str(SHAP_LABEL.get(k, k)) for k, _ in down) + ".")
    return " ".join(parts)


def facts(inc):
    """Everything the assistant may use, as plain data."""
    d = inc.data or {}
    sp = d.get("spill") or {}
    return {
        "incident": {"id": inc.id, "title": inc.title, "priority": inc.priority, "status": inc.status,
                     "area_km2": inc.area_km2, "confidence": inc.confidence,
                     "detected_at": inc.detected_at.isoformat() if inc.detected_at else None, "zones": d.get("zones", [])},
        "case": case_fields(inc),
        "spill": {k: sp.get(k) for k in ("wind_speed_ms", "reliability", "reliability_notes", "elongation",
                                         "model_disagreement", "lookalike_check", "infrastructure")},
        "detector": d.get("detector"),
        "suspects": [{"rank": s.rank, "name": s.vessel_name, "mmsi": s.mmsi, "type": s.vessel_type, "score": s.score,
                      "evidence": s.evidence, "shap": (s.features or {}).get("shap"), "review": s.review_status}
                     for s in inc.suspects[:8]],
        "dark_targets": [t for t in ((d.get("ships") or {}).get("targets") or []) if t.get("dark") or t.get("fixed")][:20],
        "forecast": [{"hours": f.get("hours"), "area_km2": f.get("area_km2"), "centroid": f.get("centroid")}
                     for f in (d.get("forecast") or [])],
        "impact": d.get("impact"),
        "decision": d.get("decision"),
    }


def rule_answer(inc, q):
    d = inc.data or {}
    sp = d.get("spill") or {}
    intent = _intent(q)
    sus = list(inc.suspects)
    if intent in ("suspects", "why"):
        if not sus:
            return "No vessel was close enough to be a suspect (or no AIS data was loaded for this image).", ["suspects"]
        named = [s for s in sus if s.vessel_name and s.vessel_name.lower() in q.lower()]
        if named:
            return _why(named[0]), ["suspects", "SHAP values"]
        if intent == "why" or re.search(r"\b(top|most|first|main)\b", q.lower()):
            share = ""
            if len(sus) > 3 and sus[0].score - sus[3].score < 0.05:
                share = (" Note: the top scores are very close, so suspicion is spread across several vessels; "
                         "no single ship stands out. This can mean a non-ship source.")
            return _why(sus[0]) + share, ["suspects", "SHAP values"]
        lines = [f"#{s.rank} {s.vessel_name} ({s.vessel_type}), score {s.score:.2f}" for s in sus[:5]]
        return f"{len(sus)} suspect vessels. Top 5: " + "; ".join(lines) + ".", ["suspects"]
    if intent == "dark":
        t = (d.get("ships") or {}).get("targets") or []
        dark = [x for x in t if x.get("dark")]
        fixed = [x for x in t if x.get("fixed")]
        msg = f"The radar image shows {len(t)} bright targets: {len(dark)} with no matching AIS signal (possible dark vessels)"
        msg += f" and {len(fixed)} fixed structures (likely oil platforms)." if fixed else "."
        if dark:
            msg += " Dark targets: " + "; ".join(f"{x['lat']:.4f}, {x['lon']:.4f}" for x in dark[:5]) + "."
        return msg, ["ships seen by radar"]
    if intent == "forecast":
        fc = d.get("forecast") or []
        if not fc:
            return "No drift forecast is stored for this incident.", ["forecast"]
        m = re.search(r"(\d+)\s*(h|hour)", q.lower())
        want = int(m.group(1)) if m else 24
        f = min(fc, key=lambda x: abs(x["hours"] - want))
        c = f.get("centroid")
        where = f" centred near {c[1]:.4f}, {c[0]:.4f}" if c else ""
        land = (d.get("impact") or {}).get("first_landfall_h")
        msg = f"After {f['hours']} h the oil is predicted to cover about {f.get('area_km2', 0):.1f} km²{where}."
        msg += f" First landfall of the main slick after about {land} h." if land is not None else " It is not expected to reach the coast within 72 h."
        return msg, ["forecast", "impact"]
    if intent == "impact":
        im = d.get("impact") or {}
        sites = im.get("threatened_sites") or []
        if not sites:
            return "No listed sensitive site is reached within 72 hours.", ["impact"]
        lines = [f"{s['name']} ({s['type']}, " + ("already within reach" if not s.get("eta_h") else f"in about {s['eta_h']} h") + ")"
                 for s in sites[:6]]
        return (f"{len(sites)} sensitive site(s) threatened: " + "; ".join(lines) +
                f". About {im.get('coastline_affected_km', 0)} km of coastline may be affected."), ["impact"]
    if intent == "size":
        im = d.get("impact") or {}
        v = im.get("volume") or {}
        msg = f"The detected slick covers {inc.area_km2:.2f} km²."
        if v:
            msg += (f" Estimated volume {v['m3'][0]}–{v['m3'][1]} m³ ({v['barrels'][0]}–{v['barrels'][1]} barrels); "
                    "the range is wide because radar cannot measure oil thickness.")
        if im.get("severity_score") is not None:
            msg += f" Severity score {im['severity_score']}/100."
        return msg, ["spill", "impact"]
    if intent == "reliability":
        la = sp.get("lookalike_check") or {}
        msg = f"Detection confidence {inc.confidence:.2f}" if inc.confidence is not None else "Detection confidence unknown"
        if sp.get("model_disagreement") is not None:
            msg += f", model disagreement {sp['model_disagreement']:.2f}"
        if sp.get("wind_speed_ms") is not None:
            msg += f", wind {sp['wind_speed_ms']} m/s"
        msg += f". Reliability: {sp.get('reliability', 'normal')}."
        if la:
            msg += f" Second-stage look-alike check: {la['verdict']} (oil score {la['oil_score']})."
            top = (la.get("evidence") or [])[:2]
            if top:
                msg += " Main reasons: " + " ".join(e["text"] for e in top)
        notes = sp.get("reliability_notes") or []
        if notes:
            msg += " Warnings: " + " ".join(notes)
        return msg, ["spill", "look-alike check"]
    if intent == "actions":
        dec = d.get("decision") or {}
        acts = dec.get("actions") or []
        msg = f"Priority {dec.get('priority', inc.priority)} (score {dec.get('priority_score', inc.priority_score)})."
        if acts:
            msg += " Recommended actions: " + " ".join(f"{i + 1}. {a['action']}." for i, a in enumerate(acts[:6]))
        return msg, ["decision support"]
    if intent == "case":
        c = case_fields(inc)
        msg = f"Status: {inc.status.replace('_', ' ')}."
        msg += f" Assigned to {c['assigned_to']['name']}." if c["assigned_to"] else " Not assigned yet."
        if c["due_at"]:
            msg += f" Deadline {c['due_at'][:16].replace('T', ' ')} UTC" + (" (OVERDUE)." if c["overdue"] else ".")
        return msg, ["case"]
    if intent == "detection":
        det = d.get("detector") or {}
        return (f"The spill was found by {det.get('model', 'the segmentation model')} on the Sentinel-1 radar image. "
                "Oil damps the waves, so it looks dark in radar; the deep models mark those dark patches, land is masked, "
                "and a second stage checks wind, contrast, edges, shape and texture."), ["detector"]
    if intent == "source":
        infra = sp.get("infrastructure") or {}
        msg = ("The spill was traced backwards with the wind and current data (brown dashed line on the map) "
               "and compared with every ship track in the hours before the image.")
        if infra.get("note"):
            msg += " " + infra["note"]
        return msg, ["back-tracking", "infrastructure"]
    # summary
    top = sus[0] if sus else None
    msg = (f"Incident #{inc.id}: {inc.area_km2:.2f} km² spill, priority {inc.priority}, status {inc.status.replace('_', ' ')}.")
    if top:
        msg += f" Top suspect {top.vessel_name} (score {top.score:.2f})."
    land = (d.get("impact") or {}).get("first_landfall_h")
    if land is not None:
        msg += f" Expected landfall after about {land} h."
    msg += " Ask me about suspects, why a ship is suspected, the forecast, threatened sites, reliability or actions."
    return msg, ["summary"]


def llm_answer(inc, q, draft):
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return None
    body = {
        "model": os.getenv("ASSISTANT_MODEL", "claude-sonnet-5-5"), "max_tokens": 500,
        "system": ("You are the analyst assistant of an oil-spill monitoring system. Answer the question in 2-5 short "
                   "sentences using ONLY the incident facts given. If the facts do not contain the answer, say so. "
                   "Vessel scores show where to investigate; they are not proof of guilt."),
        "messages": [{"role": "user", "content": f"Incident facts (JSON):\n{json.dumps(facts(inc), default=str)[:60000]}\n\n"
                                                 f"Draft answer from the rule engine:\n{draft}\n\nQuestion: {q}"}],
    }
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                 headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                          "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            out = json.loads(r.read())
        return "".join(b.get("text", "") for b in out.get("content", [])).strip() or None
    except Exception as exc:
        print(f"[assistant] LLM call failed, using rule answer: {exc}")
        return None


def answer(inc, question):
    q = (question or "").strip()[:500]
    text, used = rule_answer(inc, q)
    llm = llm_answer(inc, q, text)
    return {"answer": llm or text, "engine": "Claude" if llm else "rule-based", "sources": used,
            "suggestions": SUGGESTIONS}
