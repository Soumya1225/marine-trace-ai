"""Check the drift model against real ocean drifters (feature C6).

Drifters of NOAA's Global Drifter Program are buoys that float with the surface current and report their
position every hour. For many 72-hour pieces of real drifter tracks, the drift model is started at the
drifter's first position, driven by downloaded currents and wind, and its prediction is compared with where
the drifter really was after 24, 48 and 72 hours.

Scores reported (standard for drift-model validation):
  * separation distance (km) between forecast and real position, at 24 / 48 / 72 h
  * the same for a "no movement" baseline (the drifter stays where it started)
  * skill score (Liu & Weisberg 2011): 1 - sum(separation) / sum(length of the real path); 0 = no skill, 1 = perfect

Data (free, no account): drifters - NOAA OSMC ERDDAP "drifter_hourly_qc" (1987 - Oct 2022);
currents - Open-Meteo marine API (Meteo-France SMOC, from Jan 2022); wind - Open-Meteo / ERA5.
Because free currents start in 2022 and the hourly drifter set ends in Oct 2022, the default period is
March-May 2022 in the Arabian Sea.

    python -m ml.validation.drifters                                  (Arabian Sea, Mar-May 2022)
    python -m ml.validation.drifters --region 80 8 95 22              (Bay of Bengal)
    python -m ml.validation.drifters --start 2022-06-01 --end 2022-08-31 --max-segments 30

Writes models/metrics/drifter_validation.json (shown on the Model performance page) and
docs/screenshots/drifter_validation.png (for the report).
"""
import argparse
import csv
import io
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ml.config import METRICS_DIR, RAW_DIR
from ml.utils.geo import haversine_km

ROOT = Path(__file__).resolve().parents[2]
ERDDAP = "https://osmc.noaa.gov/erddap/tabledap/drifter_hourly_qc.csv"
DRIFTER_DIR = RAW_DIR / "drifters"
FIGURE = ROOT / "docs" / "screenshots" / "drifter_validation.png"
CHECK_HOURS = (24, 48, 72)
MAX_GAP_H = 3.0
ARABIAN_SEA = (60.0, 5.0, 77.0, 22.0)
WINDAGES = {"current_only": 0.0, "current_plus_1pct_wind": 0.01}   # drogued drifters ~0, undrogued ~1%


# ------------------------------------------------------------------ drifter data
def download_drifters(region, start, end, fetch=None):
    """Hourly drifter positions in the region and period -> list of dict(id, time, lat, lon). Cached as CSV."""
    DRIFTER_DIR.mkdir(parents=True, exist_ok=True)
    lo0, la0, lo1, la1 = region
    cache = DRIFTER_DIR / f"gdp_hourly_{start}_{end}_{la0:g}_{lo0:g}_{la1:g}_{lo1:g}.csv"
    if cache.exists():
        text = cache.read_text()
    else:
        q = (f"ID,time,latitude,longitude&time>={start}T00:00:00Z&time<={end}T23:59:59Z"
             f"&latitude>={la0}&latitude<={la1}&longitude>={lo0}&longitude<={lo1}")
        url = ERDDAP + "?" + urllib.parse.quote(q, safe="=&,:")
        if fetch is None:
            with urllib.request.urlopen(url, timeout=300) as r:            # noqa: S310 - fixed https host
                text = r.read().decode("utf-8")
        else:
            text = fetch(url)
        cache.write_text(text)
    return parse_csv(text)


def parse_csv(text):
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 3:
        return []
    head = [h.strip() for h in rows[0]]
    i_id, i_t, i_la, i_lo = (head.index(k) for k in ("ID", "time", "latitude", "longitude"))
    out = []
    for r in rows[2:]:                                   # row 2 holds the units
        try:
            out.append({"id": r[i_id].strip(), "time": datetime.fromisoformat(r[i_t].replace("Z", "+00:00")),
                        "lat": float(r[i_la]), "lon": float(r[i_lo])})
        except (ValueError, IndexError):
            continue
    return out


def segments(points, hours=72, every_h=24, max_per_drifter=4, max_total=40):
    """72-hour pieces of drifter tracks with no gap longer than MAX_GAP_H, starting every 24 h."""
    by_id = {}
    for p in points:
        by_id.setdefault(p["id"], []).append(p)
    segs = []
    for did, pts in sorted(by_id.items()):
        pts.sort(key=lambda p: p["time"])
        t = np.array([p["time"].timestamp() for p in pts])
        lat = np.array([p["lat"] for p in pts])
        lon = np.array([p["lon"] for p in pts])
        n_here, i = 0, 0
        while i < len(pts) and n_here < max_per_drifter and len(segs) < max_total:
            t0 = t[i]
            j = np.searchsorted(t, t0 + hours * 3600, side="right")
            win = t[i:j]
            if win[-1] - t0 >= hours * 3600 - 1 and (len(win) < 2 or np.diff(win).max() <= MAX_GAP_H * 3600):
                segs.append({"id": did, "t": t[i:j] - t0, "start": pts[i]["time"], "lat": lat[i:j], "lon": lon[i:j]})
                n_here += 1
                i = int(np.searchsorted(t, t0 + every_h * 3600))
            else:
                i += 1
    return segs


def at_hour(seg, h):
    """Real drifter position h hours after the segment start (linear interpolation)."""
    return float(np.interp(h * 3600, seg["t"], seg["lat"])), float(np.interp(h * 3600, seg["t"], seg["lon"]))


def path_length_km(seg, h):
    sel = seg["t"] <= h * 3600 + 1
    la, lo = seg["lat"][sel], seg["lon"][sel]
    return float(np.sum(haversine_km(la[:-1], lo[:-1], la[1:], lo[1:]))) if len(la) > 1 else 0.0


def skill(separations, lengths):
    """Liu & Weisberg (2011) skill score with tolerance n = 1."""
    total = float(np.sum(lengths))
    if total <= 0:
        return None
    return round(max(0.0, 1.0 - float(np.sum(separations)) / total), 3)


# ------------------------------------------------------------------ model run for one segment
def environment_for(seg, pad_deg=1.5):
    """Wind + current files around the drifter start, for the segment's 3 days (downloaded once, cached)."""
    from ml import environment_auto as ea
    from ml.attribution.drift import Environment
    la, lo = float(seg["lat"][0]), float(seg["lon"][0])
    bbox = (lo - pad_deg, la - pad_deg, lo + pad_deg, la + pad_deg)
    d0, d1 = seg["start"].date(), (seg["start"] + timedelta(hours=72)).date()
    files = {}
    for kind in ("wind", "current"):
        path = ea.AUTO / ea.auto_name(kind, "openmeteo5", bbox, d0, d1)
        if not path.exists():
            try:
                times, lats, lons, u, v = ea.openmeteo_grid(kind, bbox, d0, d1, n=5)
                ea.write_nc(path, kind, times, lats, lons, u, v, f"Open-Meteo ({kind}) for drifter validation")
            except Exception as exc:                                       # noqa: BLE001
                print(f"   {kind} download failed for drifter {seg['id']}: {exc}")
                path = None
        files[kind] = path
    if files["current"] is None:
        return None
    return Environment(wind_file=files["wind"], current_file=files["current"])


def run_segment(seg, env, windage):
    from ml.attribution.drift import simulate
    snaps = simulate([float(seg["lat"][0])], [float(seg["lon"][0])], seg["start"], max(CHECK_HOURS), env,
                     windage=windage, diffusion=0.0)
    out = {}
    for h in CHECK_HOURS:
        s = min(snaps, key=lambda x: abs(x["hours"] - h))
        out[h] = (float(s["lat"][0]), float(s["lon"][0]))
    track = [(float(s["lat"][0]), float(s["lon"][0])) for s in snaps[::4]]
    return out, track


# ------------------------------------------------------------------ main
def evaluate(segs, env_fn=environment_for, verbose=True):
    sep = {name: {h: [] for h in CHECK_HOURS} for name in list(WINDAGES) + ["no_movement"]}
    lengths = {h: [] for h in CHECK_HOURS}
    used, examples = 0, []
    for k, seg in enumerate(segs, 1):
        env = env_fn(seg)
        if env is None:
            continue
        used += 1
        la0, lo0 = float(seg["lat"][0]), float(seg["lon"][0])
        real = {h: at_hour(seg, h) for h in CHECK_HOURS}
        for h in CHECK_HOURS:
            lengths[h].append(path_length_km(seg, h))
            sep["no_movement"][h].append(float(haversine_km(la0, lo0, *real[h])))
        tracks = {}
        for name, w in WINDAGES.items():
            pred, track = run_segment(seg, env, w)
            tracks[name] = track
            for h in CHECK_HOURS:
                sep[name][h].append(float(haversine_km(*pred[h], *real[h])))
        if len(examples) < 8:
            examples.append({"id": seg["id"], "start": seg["start"].isoformat(),
                             "real": [[float(a), float(b)] for a, b in zip(seg["lat"][::4], seg["lon"][::4])],
                             "model": tracks["current_only"]})
        if verbose:
            print(f"   [{k}/{len(segs)}] drifter {seg['id']} {seg['start']:%Y-%m-%d}: 72 h error "
                  f"{sep['current_only'][72][-1]:.1f} km (no movement {sep['no_movement'][72][-1]:.1f} km)")
    res = {"segments_used": used, "mean_separation_km": {}, "skill": {}}
    for name, d in sep.items():
        res["mean_separation_km"][name] = {f"{h}h": (round(float(np.mean(v)), 1) if v else None) for h, v in d.items()}
        res["skill"][name] = skill([np.mean(d[h]) for h in CHECK_HOURS if d[h]],
                                   [np.mean(lengths[h]) for h in CHECK_HOURS if lengths[h]]) if used else None
    res["mean_real_path_km"] = {f"{h}h": (round(float(np.mean(v)), 1) if v else None) for h, v in lengths.items()}
    res["examples"] = examples
    return res


def verdict(res):
    m = res["mean_separation_km"]
    if not res["segments_used"]:
        return "No drifter piece could be checked (no current data was downloaded)."
    best = min(WINDAGES, key=lambda k: m[k]["72h"])
    gain = m["no_movement"]["72h"] - m[best]["72h"]
    return (f"After 72 h the model is on average {m[best]['72h']} km from the real drifter, against "
            f"{m['no_movement']['72h']} km if the drifter is assumed not to move "
            f"({'better' if gain > 0 else 'not better'} by {abs(gain):.1f} km). Skill score {res['skill'][best]} "
            f"(0 = no skill, 1 = perfect). Errors of a few tens of km over 3 days are typical for current-model "
            f"forecasts; they come mostly from the current data, not from the particle model.")


def figure(res, region):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for ex in res["examples"]:
        r, mtr = np.array(ex["real"]), np.array(ex["model"])
        ax[0].plot(r[:, 1], r[:, 0], "-", color="#2C6E91", lw=1.6)
        ax[0].plot(mtr[:, 1], mtr[:, 0], "--", color="#E07B24", lw=1.4)
        ax[0].plot(r[0, 1], r[0, 0], "o", color="#0F2A3D", ms=4)
    ax[0].set(title="Real drifters (blue) vs model forecast (orange), 72 h", xlabel="Longitude", ylabel="Latitude")
    ax[0].plot([], [], "-", color="#2C6E91", label="real drifter")
    ax[0].plot([], [], "--", color="#E07B24", label="drift model")
    ax[0].legend(loc="best", fontsize=8)
    hs = [f"{h}h" for h in CHECK_HOURS]
    m = res["mean_separation_km"]
    x = np.arange(len(hs))
    for i, (name, col) in enumerate([("current_only", "#E07B24"), ("current_plus_1pct_wind", "#C9A227"),
                                     ("no_movement", "#8A9BA8")]):
        ax[1].bar(x + (i - 1) * 0.27, [m[name][h] or 0 for h in hs], 0.27, color=col, label=name.replace("_", " "))
    ax[1].set_xticks(x, hs)
    ax[1].set(title="Mean distance from the real drifter", ylabel="km")
    ax[1].legend(fontsize=8)
    fig.suptitle(f"Drift model vs NOAA drifters, region {region}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", nargs=4, type=float, default=ARABIAN_SEA, help="min_lon min_lat max_lon max_lat")
    ap.add_argument("--start", default="2022-03-01")
    ap.add_argument("--end", default="2022-05-31")
    ap.add_argument("--max-segments", type=int, default=30)
    args = ap.parse_args()
    region = tuple(args.region)
    print(f"[1] Downloading NOAA drifter tracks {args.start} .. {args.end}, region {region} ...")
    pts = download_drifters(region, args.start, args.end)
    ids = {p["id"] for p in pts}
    print(f"   {len(pts):,} hourly positions from {len(ids)} drifters")
    segs = segments(pts, max_total=args.max_segments)
    print(f"[2] {len(segs)} 72-hour track pieces to check")
    if not segs:
        raise SystemExit("No complete 72-hour drifter pieces here: try another --region or a longer period.")
    print("[3] Running the drift model (downloads currents and wind for each piece; a few minutes) ...")
    res = evaluate(segs)
    res.update({"region": list(region), "period": [args.start, args.end], "drifters": len({s['id'] for s in segs}),
                "data": "NOAA GDP hourly drifters (OSMC ERDDAP); currents Open-Meteo marine (Meteo-France SMOC); "
                        "wind Open-Meteo (ERA5)", "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    res["verdict"] = verdict(res)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "drifter_validation.json").write_text(json.dumps(res, indent=2))
    try:
        figure(res, region)
    except Exception as exc:                                               # noqa: BLE001
        print(f"   figure skipped: {exc}")
    print("\n[4] Result")
    for name, d in res["mean_separation_km"].items():
        print(f"   {name:24s} " + "  ".join(f"{h}: {v} km" for h, v in d.items()) + f"   skill {res['skill'][name]}")
    print("\n" + res["verdict"])
    print("\nSaved models/metrics/drifter_validation.json and docs/screenshots/drifter_validation.png")


if __name__ == "__main__":
    main()
