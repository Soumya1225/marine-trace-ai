from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

METADATA = (
    ROOT
    / "Datasets"
    / "processed"
    / "dartis_2019"
    / "dartis_2019_metadata.csv"
)

DATA_ROOT = (
    ROOT
    / "Datasets"
    / "raw"
    / "dartis_2019"
)

REPORT = (
    ROOT
    / "Datasets"
    / "processed"
    / "dartis_2019"
    / "validation_report.csv"
)


SUBSET_FOLDERS = {
    "ow": DATA_ROOT / "oil" / "water",
    "oc": DATA_ROOT / "oil" / "coast",
    "nw": DATA_ROOT / "no_oil" / "water",
    "nc": DATA_ROOT / "no_oil" / "coast",
}


def validate_xml(xml_path: Path, width: int, height: int):
    errors = []
    object_count = 0

    try:
        root = ET.parse(xml_path).getroot()
    except Exception as exc:
        return 0, [f"xml_parse_error:{exc}"]

    xml_width = root.findtext("size/width")
    xml_height = root.findtext("size/height")

    try:
        xml_width = int(xml_width)
        xml_height = int(xml_height)
    except (TypeError, ValueError):
        errors.append("invalid_xml_dimensions")
    else:
        if xml_width != width or xml_height != height:
            errors.append(
                f"dimension_mismatch:{xml_width}x{xml_height}"
                f"_vs_{width}x{height}"
            )

    for obj in root.findall("object"):
        object_count += 1

        class_name = (obj.findtext("name") or "").strip().lower()

        if class_name != "oil":
            errors.append(f"unexpected_class:{class_name}")

        box = obj.find("bndbox")

        if box is None:
            errors.append("missing_bndbox")
            continue

        try:
            xmin = int(float(box.findtext("xmin", "0")))
            ymin = int(float(box.findtext("ymin", "0")))
            xmax = int(float(box.findtext("xmax", "0")))
            ymax = int(float(box.findtext("ymax", "0")))
        except ValueError:
            errors.append("invalid_bbox")
            continue

        if not (0 <= xmin < xmax <= width):
            errors.append(f"invalid_x:{xmin},{xmax}")

        if not (0 <= ymin < ymax <= height):
            errors.append(f"invalid_y:{ymin},{ymax}")

    return object_count, errors


def main():
    print("=" * 72)
    print("MarineTraceAI - DARTIS DATASET VALIDATION")
    print("=" * 72)

    if not METADATA.exists():
        raise FileNotFoundError(
            f"Metadata not found:\n{METADATA}"
        )

    df = pd.read_csv(
        METADATA,
        dtype=str,
    )

    unique_df = df.drop_duplicates(
        subset=["jpg_file"]
    ).copy()

    print(f"Metadata rows       : {len(df)}")
    print(f"Unique images       : {len(unique_df)}")
    print()

    results = []

    total = len(unique_df)

    for index, row in unique_df.reset_index(drop=True).iterrows():

        image_set = str(row["image_set"]).strip().lower()
        jpg_name = Path(
            str(row["jpg_file"]).strip()
        ).name

        xml_raw = str(
            row["xml_file"]
        ).strip()

        xml_name = (
            Path(xml_raw).name
            if xml_raw and xml_raw.lower() != "nan"
            else ""
        )

        subset_folder = SUBSET_FOLDERS.get(
            image_set
        )

        errors = []
        objects = 0
        width = None
        height = None

        if subset_folder is None:
            errors.append(
                f"unknown_subset:{image_set}"
            )
            image_path = None
            xml_path = None
        else:
            image_path = (
                subset_folder / jpg_name
            )

            xml_path = (
                subset_folder / xml_name
                if xml_name
                else None
            )

        # -------------------------
        # IMAGE VALIDATION
        # -------------------------

        if image_path is None or not image_path.exists():

            errors.append("image_missing")

        else:

            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    img.verify()

                # Re-open after verify()
                with Image.open(image_path) as img:
                    width, height = img.size

            except Exception as exc:
                errors.append(
                    f"image_invalid:{exc}"
                )

        # -------------------------
        # XML VALIDATION
        # -------------------------

        if image_set in {"ow", "oc"}:

            if xml_path is None or not xml_path.exists():

                errors.append("xml_missing")

            elif width is not None:

                objects, xml_errors = validate_xml(
                    xml_path,
                    width,
                    height,
                )

                errors.extend(xml_errors)

        results.append(
            {
                "jpg_file": jpg_name,
                "xml_file": xml_name,
                "image_set": image_set,
                "width": width,
                "height": height,
                "objects": objects,
                "valid": len(errors) == 0,
                "errors": " | ".join(errors),
            }
        )

        # Progress every 100 files
        if (index + 1) % 100 == 0 or index + 1 == total:
            print(
                f"Validated {index + 1}/{total}"
            )

    report = pd.DataFrame(results)

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report.to_csv(
        REPORT,
        index=False,
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("VALIDATION COMPLETE")
    print("=" * 72)

    valid_count = int(
        report["valid"].sum()
    )

    invalid_count = (
        len(report) - valid_count
    )

    print(
        f"Valid images   : {valid_count}"
    )

    print(
        f"Invalid images : {invalid_count}"
    )

    print(
        f"Oil objects    : "
        f"{int(report['objects'].sum())}"
    )

    print()
    print("Subset counts:")
    print(
        report["image_set"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Dimensions:")
    print(
        report[
            ["width", "height"]
        ]
        .value_counts()
        .to_string()
    )

    if invalid_count > 0:

        print()
        print("Invalid examples:")

        print(
            report[
                ~report["valid"]
            ][
                [
                    "jpg_file",
                    "image_set",
                    "errors",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    print()
    print("Report:")
    print(REPORT)


if __name__ == "__main__":
    main()
