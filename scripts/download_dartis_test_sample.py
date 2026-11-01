from pathlib import Path
from urllib.parse import quote
import requests
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_FILE = (
    PROJECT_ROOT
    / "Datasets"
    / "processed"
    / "dartis_2019"
    / "dartis_2019_metadata.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "Datasets"
    / "raw"
    / "dartis_2019"
    / "test_sample"
)

BASE_URL = "https://download.pangaea.de/dataset/980773/files/"


def download(url: str, destination: Path) -> None:
    print(f"Downloading: {url}")

    response = requests.get(
        url,
        timeout=120,
        headers={
            "User-Agent": "MarineTraceAI/1.0 academic research project"
        },
    )

    response.raise_for_status()

    destination.write_bytes(response.content)

    print(
        f"Saved: {destination.name} "
        f"({destination.stat().st_size / 1024:.2f} KB)"
    )


def main() -> None:
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(METADATA_FILE)

    print("=" * 72)
    print("MarineTraceAI - DARTIS Single Sample Test")
    print("=" * 72)

    required = ["jpg_file", "xml_file"]

    for column in required:
        if column not in df.columns:
            raise RuntimeError(
                f"Required column missing: {column}"
            )

    # Choose the first oil-water sample.
    sample = df[
        df["image_set"].astype(str).str.strip().eq("ow")
    ].iloc[0]

    jpg_file = str(sample["jpg_file"]).strip()
    xml_file = str(sample["xml_file"]).strip()

    print()
    print("Selected sample:")
    print("Image:", jpg_file)
    print("XML  :", xml_file)
    print("Patch:", sample.get("patch_name"))
    print("Date :", sample.get("start_time"))
    print("SAR  :", sample.get("sentinel_id"))
    print()

    jpg_url = BASE_URL + quote(jpg_file)
    xml_url = BASE_URL + quote(xml_file)

    jpg_output = OUTPUT_DIR / Path(jpg_file).name
    xml_output = OUTPUT_DIR / Path(xml_file).name

    download(jpg_url, jpg_output)
    download(xml_url, xml_output)

    print()
    print("=" * 72)
    print("TEST SAMPLE DOWNLOAD SUCCESSFUL")
    print("=" * 72)
    print("Image:", jpg_output)
    print("XML  :", xml_output)


if __name__ == "__main__":
    main()
