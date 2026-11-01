"""Download Natural Earth 1:10m coastline AND land polygons for the study area.

    python scripts/download_coastline.py
    python scripts/download_coastline.py --bbox -119.5 32.5 -117.0 34.5
Creates Datasets/reference/coastline.geojson (lines, used for landfall) and
Datasets/reference/land.geojson (polygons, used to stop detections on land).
"""
import argparse
import urllib.request
from pathlib import Path

LAYERS = {
    "coastline": "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip",
    "land": "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip",
}
RAW = Path("Datasets/raw/coastline")
REF = Path("Datasets/reference")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, default=(-119.5, 32.5, -117.0, 34.5))
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    REF.mkdir(parents=True, exist_ok=True)

    import geopandas as gpd
    from shapely.geometry import box
    for name, url in LAYERS.items():
        zip_path = RAW / Path(url).name
        if not zip_path.exists():
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, zip_path)
        layer = gpd.read_file(f"zip://{zip_path.resolve().as_posix()}")
        clipped = gpd.clip(layer, box(*args.bbox))
        clipped = clipped[~clipped.is_empty][["geometry"]]
        out = REF / f"{name}.geojson"
        clipped.to_file(out, driver="GeoJSON")
        print(f"Saved {len(clipped)} {name} features -> {out}")
