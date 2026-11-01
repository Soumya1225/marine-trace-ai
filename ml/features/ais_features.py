"""Vessel track helpers and behaviour features from cleaned AIS data."""
import numpy as np
import pandas as pd

from ml.config import AIS_GAP_FLAG_H, LOITER_SPEED_KN

# AIS ship type code -> (label, risk weight for oil discharge)
def vessel_type_info(code):
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "Unknown", 0.5
    if 80 <= code <= 89:
        return "Tanker", 1.0
    if 70 <= code <= 79:
        return "Cargo", 0.7
    if 60 <= code <= 69:
        return "Passenger", 0.5
    if code == 30:
        return "Fishing", 0.4
    if code in (31, 32, 52):
        return "Tug / towing", 0.4
    if code in (33, 34):
        return "Dredging / diving", 0.4
    if code in (35, 55):
        return "Military / law enforcement", 0.1
    if code in (36, 37):
        return "Pleasure craft", 0.2
    if 40 <= code <= 59:
        return "Service vessel", 0.3
    return "Other", 0.3


def tracks(df):
    """dict mmsi -> DataFrame sorted by time."""
    return {m: g.sort_values("time").reset_index(drop=True) for m, g in df.groupby("mmsi")}


def interpolate_position(track, when):
    """Linear interpolation of (lat, lon) at a timestamp; None if outside the track."""
    t = track.time.astype("int64").values / 1e9
    ts = pd.Timestamp(when).value / 1e9
    if ts < t[0] or ts > t[-1]:
        return None
    return float(np.interp(ts, t, track.lat.values)), float(np.interp(ts, t, track.lon.values))


def behaviour_features(track, focus_time=None, focus_window_h=3.0):
    """AIS gaps, slow-downs and loitering, optionally only around focus_time."""
    tr = track
    if focus_time is not None and len(track) > 1:
        ft = pd.Timestamp(focus_time)
        w = pd.Timedelta(hours=focus_window_h)
        sub = track[(track.time >= ft - w) & (track.time <= ft + w)]
        if len(sub) >= 2:
            tr = sub
    dt_h = tr.time.diff().dt.total_seconds().fillna(0).values / 3600.0
    sog = tr.sog.values.astype(float)
    moving = np.nan_to_num(sog, nan=0.0) > 0.5
    max_gap = float(dt_h.max()) if len(dt_h) else 0.0
    overall_speed = float(np.nanmedian(track.sog.values)) if track.sog.notna().any() else 0.0
    local_min = float(np.nanmin(sog)) if np.isfinite(sog).any() else overall_speed
    speed_ratio = local_min / overall_speed if overall_speed > 0.5 else 1.0
    loiter_h = float(dt_h[(np.nan_to_num(sog, nan=99) < LOITER_SPEED_KN)].sum())
    return {
        "max_gap_h": round(max_gap, 2),
        "ais_gap_flag": bool(max_gap > AIS_GAP_FLAG_H and moving.any()),
        "speed_ratio": round(float(np.clip(speed_ratio, 0, 2)), 3),
        "slowdown_flag": bool(speed_ratio < 0.5),
        "loiter_h": round(loiter_h, 2),
        "median_speed_kn": round(overall_speed, 2),
    }


def vessel_summary(track):
    last = track.iloc[-1]
    label, risk = vessel_type_info(last.get("vessel_type"))
    return {
        "mmsi": int(last["mmsi"]),
        "vessel_name": str(last.get("vessel_name") or "") or f"MMSI {int(last['mmsi'])}",
        "imo": None if pd.isna(last.get("imo")) else str(last.get("imo")),
        "vessel_type_code": None if pd.isna(last.get("vessel_type")) else int(last.get("vessel_type")),
        "vessel_type": label,
        "type_risk": risk,
        "length_m": None if pd.isna(last.get("length")) else float(last.get("length")),
    }


def track_to_json(track, max_points=400):
    step = max(1, len(track) // max_points)
    t = track.iloc[::step]
    return [{"t": ts.isoformat(), "lat": round(float(la), 6), "lon": round(float(lo), 6),
             "sog": None if pd.isna(s) else round(float(s), 1), "cog": None if pd.isna(c) else round(float(c), 1)}
            for ts, la, lo, s, c in zip(t.time, t.lat, t.lon, t.sog, t.cog)]
