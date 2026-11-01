from pathlib import Path
import requests

DOI_URL = "https://doi.pangaea.de/10.1594/PANGAEA.980773?format=textfile"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "Datasets" / "raw" / "dartis_2019"
OUTPUT_FILE = OUTPUT_DIR / "DARTIS_2019.tab"


def download_file() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MarineTraceAI - DARTIS 2019 Metadata Downloader")
    print("=" * 72)
    print(f"Source : {DOI_URL}")
    print(f"Output : {OUTPUT_FILE}")
    print()

    response = requests.get(
        DOI_URL,
        timeout=120,
        headers={
            "User-Agent": "MarineTraceAI/1.0 academic research project"
        },
    )

    response.raise_for_status()

    OUTPUT_FILE.write_bytes(response.content)

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)

    print("Download successful.")
    print(f"Saved file : {OUTPUT_FILE}")
    print(f"Size       : {size_mb:.2f} MB")


if __name__ == "__main__":
    download_file()
