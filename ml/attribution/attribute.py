"""Full attribution for one spill: back-track drift -> candidate vessels -> features -> ranking."""
import pandas as pd

from ml.attribution.candidate_filter import backtrack_offsets, candidate_vessels
from ml.attribution.drift import centroid_track, seed_particles, simulate
from ml.attribution.scorer import rank_vessels
from ml.config import ATTRIBUTION_WINDOW_H, RESPONSE_NAME_KEYWORDS, RESPONSE_TYPE_CODES, SEARCH_RADIUS_KM


def split_responders(cands):
    """Separate emergency / response vessels from real candidates."""
    if cands is None or cands.empty:
        return cands, []
    last = cands.sort_values("time").groupby("mmsi").tail(1)
    names = last.vessel_name.fillna("").astype(str).str.upper()
    codes = pd.to_numeric(last.vessel_type, errors="coerce")
    is_resp = codes.isin(RESPONSE_TYPE_CODES) | names.apply(lambda n: any(k in n for k in RESPONSE_NAME_KEYWORDS))
    excluded = [{"mmsi": int(m), "vessel_name": n or f"MMSI {int(m)}"} for m, n in zip(last.mmsi[is_resp], last.vessel_name[is_resp])]
    return cands[~cands.mmsi.isin(last.mmsi[is_resp])], excluded


def backtrack(spill, image_time, env, hours=ATTRIBUTION_WINDOW_H, n=200):
    lat, lon = seed_particles(spill["axis"], n=n)
    snaps = simulate(lat, lon, pd.Timestamp(image_time).to_pydatetime(), hours, env, direction=-1)
    return centroid_track(snaps)


def attribute_spill(spill, ais, image_time, env, ranker=None, window_h=ATTRIBUTION_WINDOW_H,
                    radius_km=SEARCH_RADIUS_KM, top_k=10, include_tracks=True, context=None):
    hours, blat, blon = backtrack(spill, image_time, env, hours=window_h)
    offset = backtrack_offsets(hours, blat, blon)
    cands = candidate_vessels(ais, spill, image_time, offset, window_h, radius_km)
    cands, excluded = split_responders(cands)
    spill["excluded_responders"] = excluded
    suspects = rank_vessels(cands, spill, offset, ranker=ranker, top_k=top_k, include_tracks=include_tracks,
                            context=context)
    backtrack_path = [[round(float(lo), 6), round(float(la), 6), round(float(h), 2)]
                      for h, la, lo in zip(hours, blat, blon)]
    return suspects, backtrack_path
