from pathlib import Path
from urllib.parse import quote
import time
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

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "Datasets"
    / "raw"
    / "dartis_2019"
)

BASE_URL = "https://download.pangaea.de/dataset/980773/files/"

MANIFEST_FILE = (
    PROJECT_ROOT
    / "Datasets"
    / "processed"
    / "dartis_2019"
    / "download_manifest.csv"
)


SUBSET_TO_FOLDER = {
    "ow": Path("oil") / "water",
    "oc": Path("oil") / "coast",
    "nw": Path("no_oil") / "water",
    "nc": Path("no_oil") / "coast",
}


def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_download_plan(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    # One row per unique JPEG.
    image_rows = (
        df[
            [
                "image_set",
                "jpg_file",
                "xml_file",
                "patch_name",
                "start_time",
                "end_time",
                "sentinel_id",
            ]
        ]
        .drop_duplicates(subset=["jpg_file"])
        .copy()
    )

    for _, row in image_rows.iterrows():
        image_set = clean_value(row["image_set"]).lower()
        jpg_file = clean_value(row["jpg_file"])
        xml_file = clean_value(row["xml_file"])

        if not jpg_file:
            continue

        if image_set not in SUBSET_TO_FOLDER:
            print(f"WARNING: unknown image_set: {image_set}")
            continue

        folder = OUTPUT_ROOT / SUBSET_TO_FOLDER[image_set]

        records.append(
            {
                "image_set": image_set,
                "jpg_file": jpg_file,
                "xml_file": xml_file,
                "patch_name": clean_value(row["patch_name"]),
                "start_time": clean_value(row["start_time"]),
                "end_time": clean_value(row["end_time"]),
                "sentinel_id": clean_value(row["sentinel_id"]),
                "folder": str(folder),
            }
        )

    return pd.DataFrame(records)


def download_file(
    session: requests.Session,
    url: str,
    destination: Path,
    retries: int = 4,
) -> tuple[bool, str]:
    if destination.exists() and destination.stat().st_size > 0:
        return True, "already_exists"

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                url,
                timeout=(20, 180),
            )

            response.raise_for_status()

            temp_file = destination.with_suffix(
                destination.suffix + ".part"
            )

            temp_file.write_bytes(response.content)

            if temp_file.stat().st_size == 0:
                temp_file.unlink(missing_ok=True)
                raise RuntimeError("Downloaded file is empty.")

            temp_file.replace(destination)

            return True, "downloaded"

        except Exception as exc:
            if attempt == retries:
                return False, str(exc)

            wait_seconds = attempt * 2
            time.sleep(wait_seconds)

    return False, "unknown_error"


def main() -> None:
    print("=" * 80)
    print("MarineTraceAI - DARTIS 2019 FULL DATASET DOWNLOADER")
    print("=" * 80)

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_FILE}"
        )

    df = pd.read_csv(
        METADATA_FILE,
        dtype=str,
    )

    required_columns = {
        "image_set",
        "jpg_file",
        "xml_file",
        "patch_name",
        "start_time",
        "end_time",
        "sentinel_id",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing metadata columns: {sorted(missing)}"
        )

    plan = build_download_plan(df)

    print()
    print(f"Metadata rows          : {len(df)}")
    print(f"Unique images planned  : {plan['jpg_file'].nunique()}")
    print(
        "Unique XML annotations : "
        f"{plan['xml_file'].replace('', pd.NA).dropna().nunique()}"
    )

    print()
    print("Subset counts:")
    print(
        plan["image_set"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    manifest_rows = []

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "MarineTraceAI/1.0 "
                "academic research project"
            )
        }
    )

    total = len(plan)

    for index, row in plan.reset_index(drop=True).iterrows():
        number = index + 1

        image_set = row["image_set"]
        jpg_file = row["jpg_file"]
        xml_file = row["xml_file"]

        folder = Path(row["folder"])

        jpg_destination = folder / Path(jpg_file).name

        jpg_url = BASE_URL + quote(
            Path(jpg_file).name,
            safe="",
        )

        image_ok, image_status = download_file(
            session,
            jpg_url,
            jpg_destination,
        )

        xml_status = "not_required"
        xml_ok = True

        # XML files are available for oil images.
        if xml_file:
            xml_destination = folder / Path(xml_file).name

            xml_url = BASE_URL + quote(
                Path(xml_file).name,
                safe="",
            )

            xml_ok, xml_status = download_file(
                session,
                xml_url,
                xml_destination,
            )

        overall_ok = image_ok and xml_ok

        manifest_rows.append(
            {
                "image_set": image_set,
                "jpg_file": jpg_file,
                "xml_file": xml_file,
                "patch_name": row["patch_name"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "sentinel_id": row["sentinel_id"],
                "image_status": image_status,
                "xml_status": xml_status,
                "success": overall_ok,
            }
        )

        print(
            f"[{number:4d}/{total}] "
            f"{image_set:2s} "
            f"{jpg_file:20s} "
            f"JPG={image_status:14s} "
            f"XML={xml_status}"
        )

        # Save progress after every image.
        pd.DataFrame(manifest_rows).to_csv(
            MANIFEST_FILE,
            index=False,
            encoding="utf-8",
        )

    manifest = pd.DataFrame(manifest_rows)

    success_count = int(manifest["success"].sum())
    failed_count = len(manifest) - success_count

    print()
    print("=" * 80)
    print("DARTIS DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"Total image records : {len(manifest)}")
    print(f"Successful          : {success_count}")
    print(f"Failed              : {failed_count}")
    print()
    print(f"Manifest saved to:")
    print(MANIFEST_FILE)

    if failed_count:
        failures = manifest[
            ~manifest["success"]
        ]

        failure_file = (
            PROJECT_ROOT
            / "Datasets"
            / "processed"
            / "dartis_2019"
            / "download_errors.csv"
        )

        failures.to_csv(
            failure_file,
            index=False,
            encoding="utf-8",
        )

        print()
        print("Some downloads failed.")
        print(f"Failure list: {failure_file}")


if __name__ == "__main__":
    main()
