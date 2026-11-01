"""Track a spill across two satellite images and check the drift forecast against reality.

The first image's analysis predicts where each slick will be after N hours. A second image taken N hours
later shows where oil actually is. This script compares the two:

  * forecast position vs. the nearest slick detected in the second image (distance in km)
  * how much of the oil seen in the second image falls inside the forecast uncertainty zone
  * the same numbers for a "persistence" baseline (assume the slick did not move)
  * whether the second image covers the forecast area at all, and whether its wind was high enough
    for a reliable detection

    python -m ml.validation.track_between_images                      (newest two analyses, automatic)
    python -m ml.validation.track_between_images --first backend/uploads/results/job_1 --second .../job_2

Writes models/metrics/forecast_validation.json (shown on the Model performance page) and
docs/screenshots/forecast_validation.png (for the report).
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from ml.config import METRICS_DIR, MIN_RELIABLE_WIND_MS
from ml.utils.geo import KM_PER_DEG_LAT, haversine_km, km_per_deg_lon

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "backend" / "uploads" / "results"
FIGURE = ROOT / "docs" / "screenshots" / "forecast_validation.png"
MATCH_KM = 10.0   # a second-image slick farther than this from the forecast is a different slick


def load_run(folder):
    folder = Path(folder)
    r = json.loads((folder / "result.json").read_text())
    r["_dir"] = folder
    r["_time"] = datetime.fromisoformat(r["scene"]["acquisition_time"])
    return r


def find_runs():
    runs = [load_run(p.parent) for p in RESULTS_DIR.glob("*/result.json")]
    runs = [r for r in runs if r.get("spills") is not None]
    if len(runs) < 2:
        raise SystemExit("Need at least two finished analyses of the same area (backend/uploads/results/*/result.json)")
    runs.sort(key=lambda r: r["_time"])
    return runs[-2], runs[-1]


def inside(poly, lon, lat):
    """Boolean mask: which points (lon, lat arrays) lie inside a [[lon, lat], ...] polygon."""
    if not poly or len(poly) < 3:
        return np.zeros(np.shape(lon), bool)
    from matplotlib.path import Path as MPath
    pts = np.column_stack([np.ravel(lon), np.ravel(lat)])
    return MPath(np.asarray(poly)).contains_points(pts).reshape(np.shape(lon))


def oil_pixels(run):
    """Lon/lat of every oil pixel in a run's detection mask, the valid-data mask, and the pixel grid."""
    d = run["_dir"]
    mask = np.asarray(Image.open(d / "mask.png").convert("RGBA"))
    sar = np.asarray(Image.open(d / "sar.png").convert("L"))
    min_lon, min_lat, max_lon, max_lat = run["scene"]["bounds"]
    h, w = mask.shape[:2]
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    lon = min_lon + (cols + 0.5) / w * (max_lon - min_lon)
    lat = max_lat - (rows + 0.5) / h * (max_lat - min_lat)
    oil = (mask[..., 0] > 200) & (mask[..., 1] < 100) & (mask[..., 3] > 0)
    valid = sar > 0
    px_km2 = ((max_lon - min_lon) / w * km_per_deg_lon((min_lat + max_lat) / 2)) * ((max_lat - min_lat) / h * KM_PER_DEG_LAT)
    return lon, lat, oil, valid, px_km2


def forecast_at(forecast, hours):
    """Forecast entry nearest to `hours`, with its centre interpolated to exactly `hours`."""
    fc = sorted(forecast or [], key=lambda f: f["hours"])
    if not fc:
        return None
    near = min(fc, key=lambda f: abs(f["hours"] - hours))
    before = [f for f in fc if f["hours"] <= hours]
    after = [f for f in fc if f["hours"] >= hours]
    if before and after and before[-1]["hours"] != after[0]["hours"]:
        a, b = before[-1], after[0]
        t = (hours - a["hours"]) / (b["hours"] - a["hours"])
        lat = a["centroid_lat"] + t * (b["centroid_lat"] - a["centroid_lat"])
        lon = a["centroid_lon"] + t * (b["centroid_lon"] - a["centroid_lon"])
    else:
        lat, lon = near["centroid_lat"], near["centroid_lon"]
    return {"hours_used": near["hours"], "centroid": (lat, lon), "polygon": near.get("polygon"),
            "uncertainty_polygon": near.get("uncertainty_polygon") or near.get("polygon"),
            "lstm_centroid": near.get("lstm_centroid")}


def compare(first, second):
    gap_h = (second["_time"] - first["_time"]).total_seconds() / 3600
    if gap_h <= 0:
        raise SystemExit("The second image must be taken after the first one")
    lon2, lat2, oil2, valid2, px_km2 = oil_pixels(second)
    second_spills = [s["spill"] for s in second["spills"]]
    wind2 = second.get("scene_wind_ms")
    oil_total_km2 = float(oil2.sum() * px_km2)

    rows = []
    for item in first["spills"]:
        sp = item["spill"]
        fc = forecast_at(item.get("forecast"), gap_h)
        if fc is None:
            continue
        start = (sp["centroid_lat"], sp["centroid_lon"])
        zone = inside(fc["uncertainty_polygon"], lon2, lat2)
        start_zone = inside(sp.get("polygon"), lon2, lat2)

        def nearest(pt):
            if not second_spills:
                return None, None
            d = [haversine_km(pt[0], pt[1], s["centroid_lat"], s["centroid_lon"]) for s in second_spills]
            i = int(np.argmin(d))
            return round(float(d[i]), 2), second_spills[i].get("spill_no")

        d_fc, match_fc = nearest(fc["centroid"])
        d_start, _ = nearest(start)
        d_lstm = nearest(tuple(fc["lstm_centroid"]))[0] if fc.get("lstm_centroid") else None
        covered = float(valid2[zone].mean()) if zone.any() else 0.0
        rows.append({
            "spill_no": sp.get("spill_no"),
            "area_km2": sp.get("area_km2"),
            "reliability_first": sp.get("reliability"),
            "forecast_hours_used": fc["hours_used"],
            "start": [round(start[0], 5), round(start[1], 5)],
            "forecast_centre": [round(fc["centroid"][0], 5), round(fc["centroid"][1], 5)],
            "moved_km": round(float(haversine_km(*start, *fc["centroid"])), 2),
            "second_image_covers_zone": round(covered, 3),
            "nearest_second_spill": match_fc,
            "dist_forecast_km": d_fc,
            "dist_lstm_km": d_lstm,
            "dist_persistence_km": d_start,
            "oil_in_forecast_zone_km2": round(float((oil2 & zone).sum() * px_km2), 3),
            "oil_in_original_outline_km2": round(float((oil2 & start_zone).sum() * px_km2), 3),
            "_fc": fc, "_poly": sp.get("polygon"),
        })

    for r in rows:
        r["matched"] = bool(r["second_image_covers_zone"] >= 0.5 and (
            r["oil_in_forecast_zone_km2"] > 0 or (r["dist_forecast_km"] is not None and r["dist_forecast_km"] <= MATCH_KM)))
    usable = [r for r in rows if r["matched"]]
    closer = sum(1 for r in usable if r["dist_forecast_km"] < r["dist_persistence_km"])
    reliable2 = wind2 is None or wind2 >= MIN_RELIABLE_WIND_MS
    if not usable:
        covered = [r for r in rows if r["second_image_covers_zone"] >= 0.5]
        if covered:
            verdict = (f"No match: the second image covers the forecast areas but shows no oil there (nearest slick "
                       f"{min(r['dist_forecast_km'] for r in covered if r['dist_forecast_km'] is not None):.0f} km away, "
                       f"more than {MATCH_KM:.0f} km, so it is a different slick). The forecast cannot be checked with this pair.")
        else:
            verdict = "Not testable: the second image does not cover the forecast areas."
    else:
        verdict = (f"In {closer} of {len(usable)} testable spills the forecast lands closer to the oil seen "
                   f"{gap_h:.1f} h later than the 'did not move' baseline.")
    if not reliable2:
        verdict += (f" Reason: wind in the second image was only {wind2} m/s. In a flat calm the whole sea looks dark "
                    "on radar, so oil cannot be told apart from water and the detector filters these areas out. "
                    "A second image with wind above " f"{MIN_RELIABLE_WIND_MS} m/s is needed.")

    summary = {
        "first_image": first["scene"]["source"], "first_time": first["_time"].isoformat(),
        "second_image": second["scene"]["source"], "second_time": second["_time"].isoformat(),
        "gap_hours": round(gap_h, 2),
        "second_image_wind_ms": wind2, "second_image_reliable": reliable2,
        "second_image_spills": len(second_spills), "second_image_oil_km2": round(oil_total_km2, 3),
        "match_radius_km": MATCH_KM,
        "testable_spills": len(usable), "forecast_closer_than_baseline": closer,
        "mean_dist_forecast_km": round(float(np.mean([r["dist_forecast_km"] for r in usable])), 2) if usable else None,
        "mean_dist_persistence_km": round(float(np.mean([r["dist_persistence_km"] for r in usable])), 2) if usable else None,
        "verdict": verdict,
        "spills": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
    }
    return summary, rows, (lon2, lat2, oil2, second)


def figure(summary, rows, grid, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lon2, lat2, oil2, second = grid
    min_lon, min_lat, max_lon, max_lat = second["scene"]["bounds"]
    sar = np.asarray(Image.open(second["_dir"] / "sar.png").convert("L"))
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=130)
    ax.imshow(sar, cmap="gray", vmin=0, vmax=255, extent=(min_lon, max_lon, min_lat, max_lat), origin="upper")
    rgba = np.zeros(oil2.shape + (4,))
    rgba[oil2] = (0.85, 0.15, 0.1, 0.75)
    ax.imshow(rgba, extent=(min_lon, max_lon, min_lat, max_lat), origin="upper")
    for i, r in enumerate(rows):
        fc = r["_fc"]
        if fc["uncertainty_polygon"]:
            xs, ys = zip(*fc["uncertainty_polygon"])
            ax.fill(xs, ys, color="#2C6E91", alpha=0.18, label="Forecast zone" if i == 0 else None)
            ax.plot(xs + xs[:1], ys + ys[:1], color="#2C6E91", lw=0.8)
        if r["_poly"]:
            xs, ys = zip(*r["_poly"])
            ax.plot(xs + xs[:1], ys + ys[:1], color="#E8A42A", lw=1.2, label="Slick in first image" if i == 0 else None)
        (sla, slo), (fla, flo) = r["start"], r["forecast_centre"]
        ax.annotate("", (flo, fla), (slo, sla), arrowprops=dict(arrowstyle="->", color="#2C6E91", lw=1.2))
        ax.plot(flo, fla, "o", color="#2C6E91", ms=4, label="Forecast centre" if i == 0 else None)
    ax.plot([], [], "s", color=(0.85, 0.15, 0.1), label=f"Oil detected {summary['gap_hours']:.0f} h later")
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Forecast from first image vs. oil seen {summary['gap_hours']:.1f} h later", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", help="result folder of the earlier image")
    ap.add_argument("--second", help="result folder of the later image")
    args = ap.parse_args()
    if args.first and args.second:
        first, second = sorted([load_run(args.first), load_run(args.second)], key=lambda r: r["_time"])
    else:
        first, second = find_runs()
    summary, rows, grid = compare(first, second)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "forecast_validation.json").write_text(json.dumps(summary, indent=2))
    figure(summary, rows, grid, FIGURE)

    print(f"First image : {summary['first_image']}  ({summary['first_time']})")
    print(f"Second image: {summary['second_image']}  ({summary['second_time']})")
    print(f"Gap {summary['gap_hours']} h | second-image wind {summary['second_image_wind_ms']} m/s | "
          f"slicks in second image {summary['second_image_spills']} | oil area {summary['second_image_oil_km2']} km2")
    for r in summary["spills"]:
        print(f"  Spill {r['spill_no']} ({r['area_km2']} km2): moved {r['moved_km']} km, "
              f"zone covered by 2nd image {r['second_image_covers_zone'] * 100:.0f}%, "
              f"nearest 2nd-image slick: forecast {r['dist_forecast_km']} km / LSTM {r['dist_lstm_km']} km / "
              f"no-move {r['dist_persistence_km']} km, oil inside zone {r['oil_in_forecast_zone_km2']} km2, "
              f"{'MATCHED' if r['matched'] else 'no match'}")
    print(summary["verdict"])
    print(f"Saved models/metrics/forecast_validation.json and {FIGURE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
