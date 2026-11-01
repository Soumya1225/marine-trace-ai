"""Load and clean AIS CSV files (MarineCadastre US format and Danish Maritime Authority format).

Big daily files (~900 MB) are read in chunks and filtered by area and time while reading.

Usage:
    python -m ml.preprocessing.ais_clean Datasets/raw/ais/AIS_2021_10_02.csv \
        --bbox -118.6 33.3 -117.6 33.9 --out Datasets/processed/ais_huntington.pkl
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Map many possible source column names -> standard names
COLUMN_MAP = {
    "mmsi": "mmsi",
    "basedatetime": "time", "# timestamp": "time", "timestamp": "time",
    "lat": "lat", "latitude": "lat",
    "lon": "lon", "longitude": "lon",
    "sog": "sog", "cog": "cog", "heading": "heading",
    "vesselname": "vessel_name", "name": "vessel_name",
    "imo": "imo",
    "vesseltype": "vessel_type", "ship type": "vessel_type",
    "length": "length",
}
STANDARD = ["mmsi", "time", "lat", "lon", "sog", "cog", "heading",
            "vessel_name", "imo", "vessel_type", "length"]

# DMA uses text ship types; map the important ones to AIS numeric codes
DMA_TYPE_CODES = {"tanker": 80, "cargo": 70, "fishing": 30, "passenger": 60,
                  "tug": 52, "towing": 31, "pleasure": 37, "sailing": 36,
                  "military": 35, "pilot": 50, "sar": 51, "dredging": 33}


def _standardize(chunk):
    rename = {}
    for col in chunk.columns:
        key = col.strip().lower()
        if key in COLUMN_MAP and COLUMN_MAP[key] not in rename.values():
            rename[col] = COLUMN_MAP[key]
    chunk = chunk.rename(columns=rename)
    for col in STANDARD:
        if col not in chunk.columns:
            chunk[col] = np.nan
    return chunk[STANDARD]


def _vessel_type_to_code(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any() or series.isna().all():
        return numeric
    text = series.astype(str).str.lower()
    out = pd.Series(np.nan, index=series.index)
    for word, code in DMA_TYPE_CODES.items():
        out[text.str.contains(word, na=False)] = code
    return out


def _utc(t):
    if t is None:
        return None
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def load_ais(paths, bbox=None, t_start=None, t_end=None, chunksize=500_000):
    """Read one or more AIS CSVs, keep rows inside bbox and [t_start, t_end], clean them.

    bbox = (min_lon, min_lat, max_lon, max_lat); times are pandas-parsable (UTC).
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    t_start = _utc(t_start)
    t_end = _utc(t_end)
    parts = []
    for path in paths:
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            chunk = _standardize(chunk)
            chunk["lat"] = pd.to_numeric(chunk["lat"], errors="coerce")
            chunk["lon"] = pd.to_numeric(chunk["lon"], errors="coerce")
            if bbox is not None:
                min_lon, min_lat, max_lon, max_lat = bbox
                chunk = chunk[chunk.lon.between(min_lon, max_lon) & chunk.lat.between(min_lat, max_lat)]
            if chunk.empty:
                continue
            dayfirst = chunk["time"].astype(str).str.match(r"^\d{2}/\d{2}/\d{4}").any()
            chunk["time"] = pd.to_datetime(chunk["time"], errors="coerce", utc=True, dayfirst=bool(dayfirst))
            if t_start is not None:
                chunk = chunk[chunk.time >= t_start]
            if t_end is not None:
                chunk = chunk[chunk.time <= t_end]
            if not chunk.empty:
                parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=STANDARD)
    return clean_ais(pd.concat(parts, ignore_index=True))


def clean_ais(df):
    """Remove invalid MMSIs, impossible values, duplicates and position jumps."""
    df = df.copy()
    df["mmsi"] = pd.to_numeric(df["mmsi"], errors="coerce")
    df = df[df.mmsi.between(100_000_000, 999_999_999)]
    for col in ["sog", "cog", "heading", "length"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["vessel_type"] = _vessel_type_to_code(df["vessel_type"])
    df = df[df.lat.between(-90, 90) & df.lon.between(-180, 180)]
    df = df[df.time.notna()]
    df.loc[(df.sog < 0) | (df.sog > 50), "sog"] = np.nan          # 102.3 = not available
    df.loc[(df.cog < 0) | (df.cog >= 360), "cog"] = np.nan
    df.loc[(df.heading < 0) | (df.heading >= 360), "heading"] = np.nan
    df = df.drop_duplicates(["mmsi", "time"]).sort_values(["mmsi", "time"])
    df["mmsi"] = df["mmsi"].astype("int64")
    df["vessel_name"] = df["vessel_name"].fillna("").astype(str).str.strip()
    df = _remove_jumps(df)
    return df.reset_index(drop=True)


def _remove_jumps(df, max_speed_kn=60):
    """Drop points that imply an impossible speed from the previous point of the same ship."""
    from ml.utils.geo import haversine_km
    keep = np.ones(len(df), dtype=bool)
    prev_mmsi = df.mmsi.shift()
    dist = haversine_km(df.lat.shift().values, df.lon.shift().values, df.lat.values, df.lon.values)
    dt_h = (df.time - df.time.shift()).dt.total_seconds().values / 3600.0
    with np.errstate(divide="ignore", invalid="ignore"):
        speed_kn = dist / dt_h / 1.852
    same = (df.mmsi == prev_mmsi).values
    keep[same & (speed_kn > max_speed_kn) & (dt_h > 0)] = False
    return df[keep]


def main():
    ap = argparse.ArgumentParser(description="Clean and crop AIS CSV files")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out", default="Datasets/processed/ais_clean.pkl")
    args = ap.parse_args()
    df = load_ais(args.paths, args.bbox, args.start, args.end)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(args.out)
    print(f"Saved {len(df):,} AIS points from {df.mmsi.nunique():,} vessels -> {args.out}")


if __name__ == "__main__":
    main()
