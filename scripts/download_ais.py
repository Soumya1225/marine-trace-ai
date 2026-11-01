"""Download daily US AIS files from MarineCadastre (NOAA) and extract them into Datasets/raw/ais/.

    python scripts/download_ais.py 2021-10-02 2021-10-03
    python scripts/download_ais.py 2021-01-24 2021-01-25 2021-01-26
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

OUT = Path("Datasets/raw/ais")
BASE = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{stamp}.zip"


def progress(block, block_size, total):
    if total > 0:
        done = min(block * block_size, total)
        print(f"\r  {done / 1e6:6.0f} / {total / 1e6:.0f} MB", end="")


def fetch(day):
    stamp = day.replace("-", "_")
    csv_path = OUT / f"AIS_{stamp}.csv"
    if csv_path.exists():
        print(f"{csv_path.name} already exists, skipping")
        return
    url = BASE.format(year=day[:4], stamp=stamp)
    zip_path = OUT / f"AIS_{stamp}.zip"
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, zip_path, progress)
    print("\n  extracting...")
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            if member.lower().endswith(".csv"):
                with z.open(member) as src, open(OUT / Path(member).name, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
    zip_path.unlink()
    print(f"  saved {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/download_ais.py YYYY-MM-DD [YYYY-MM-DD ...]")
    OUT.mkdir(parents=True, exist_ok=True)
    for d in sys.argv[1:]:
        fetch(d)
