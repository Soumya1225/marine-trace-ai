"""Search and download Sentinel-1 GRD scenes from the Copernicus Data Space Ecosystem.

Needs CDSE_USER / CDSE_PASSWORD in the .env file (free account at dataspace.copernicus.eu).

    python ml/preprocessing/download_sentinel1.py                 # list scenes, then choose
    python ml/preprocessing/download_sentinel1.py --name 20211003T135159   # download one scene by name
    python ml/preprocessing/download_sentinel1.py --aoi -118.15 33.55 -117.95 33.68 --start 2021-09-28 --end 2021-10-10
"""
import argparse
import os

import requests
from dotenv import load_dotenv

load_dotenv()
OUT_DIR = "Datasets/raw/sentinel1"
DEFAULT_AOI = (-118.15, 33.55, -117.95, 33.68)   # Huntington Beach spill zone
DEFAULT_START, DEFAULT_END = "2021-09-28", "2021-10-10"


def search(aoi, start, end, cog_only=True):
    min_lon, min_lat, max_lon, max_lat = aoi
    poly = (f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, "
            f"{min_lon} {max_lat}, {min_lon} {min_lat}))")
    url = (
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter="
        "Collection/Name eq 'SENTINEL-1' and contains(Name,'IW_GRDH') and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{poly}') and "
        f"ContentDate/Start gt {start}T00:00:00.000Z and ContentDate/Start lt {end}T23:59:59.000Z"
        "&$orderby=ContentDate/Start&$top=50"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    items = r.json().get("value", [])
    if cog_only:
        items = [p for p in items if p["Name"].endswith("_COG.SAFE")] or items
    return items


def get_token():
    user, password = os.getenv("CDSE_USER"), os.getenv("CDSE_PASSWORD")
    if not user or not password or user.startswith("your_"):
        raise SystemExit("Put your real Copernicus email/password in the .env file first")
    r = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={"client_id": "cdse-public", "username": user, "password": password, "grant_type": "password"},
        timeout=60)
    if r.status_code != 200:
        raise SystemExit(f"Login failed ({r.status_code}): {r.text[:200]}")
    return r.json()["access_token"]


def download(product, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    session = requests.Session()
    print("Logging in to Copernicus...", flush=True)
    session.headers.update({"Authorization": f"Bearer {get_token()}"})
    url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product['Id']})/$value"
    out_path = os.path.join(out_dir, product["Name"] + ".zip")
    print("Logged in. Starting download (the server can take a minute to start sending)...", flush=True)
    # follow redirects by hand (the token must be kept), streaming from the first request so the
    # 900 MB file is downloaded only once and progress shows straight away
    r = session.get(url, allow_redirects=False, stream=True, timeout=120)
    while r.status_code in (301, 302, 303, 307):
        url = r.headers["Location"]
        r.close()
        r = session.get(url, allow_redirects=False, stream=True, timeout=120)
    with r:
        r.raise_for_status()
        total, done = int(r.headers.get("content-length", 0)), 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\rDownloaded {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
                else:
                    print(f"\rDownloaded {done / 1e6:.0f} MB", end="", flush=True)
    print(f"\nSaved: {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", nargs=4, type=float, default=DEFAULT_AOI)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--index", type=int, help="download this scene number without asking")
    ap.add_argument("--name", help="download the scene whose name contains this text, e.g. 20211003T135159")
    ap.add_argument("--all-formats", action="store_true", help="also list non-COG duplicates")
    args = ap.parse_args()
    products = search(args.aoi, args.start, args.end, cog_only=not args.all_formats)
    if not products:
        raise SystemExit("No scenes found. Try a wider date range.")
    print(f"Found {len(products)} scenes:\n")
    for i, p in enumerate(products):
        print(f"[{i}] {p['ContentDate']['Start'][:19]}  {p['Name']}")
    if args.name:
        match = [p for p in products if args.name in p["Name"]]
        if not match:
            raise SystemExit(f"No scene name contains '{args.name}'")
        download(match[0])
        return
    choice = args.index if args.index is not None else input("\nScene number to download (Enter = skip): ").strip()
    if str(choice).isdigit() and int(choice) < len(products):
        download(products[int(choice)])
    else:
        print("Skipped download.")


if __name__ == "__main__":
    main()
