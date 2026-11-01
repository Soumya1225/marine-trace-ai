from pathlib import Path
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]

METADATA = (
    ROOT
    / "Datasets"
    / "processed"
    / "dartis_2019"
    / "dartis_2019_metadata.csv"
)

VALIDATION_REPORT = (
    ROOT
    / "Datasets"
    / "processed"
    / "dartis_2019"
    / "validation_report.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Datasets"
    / "processed"
    / "dartis_2019"
    / "splits"
)


def main():
    print("=" * 72)
    print("MarineTraceAI - DARTIS SCENE-LEVEL DATASET SPLIT")
    print("=" * 72)

    df = pd.read_csv(METADATA, dtype=str)

    # One row per unique image.
    df = df.drop_duplicates(
        subset=["jpg_file"]
    ).copy()

    df["image_set"] = (
        df["image_set"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["sentinel_id"] = (
        df["sentinel_id"]
        .astype(str)
        .str.strip()
    )

    # Binary class for detection.
    df["class"] = df["image_set"].isin(
        ["ow", "oc"]
    ).astype(int)

    # Group by Sentinel-1 acquisition/product.
    groups = df["sentinel_id"]

    print()
    print("Unique images        :", len(df))
    print("Unique Sentinel IDs  :", groups.nunique())
    print("Oil images           :", int(df["class"].sum()))
    print("No-oil images        :", int((df["class"] == 0).sum()))

    # ----------------------------------------------------------
    # First split: 70% train / 30% temporary
    # ----------------------------------------------------------

    gss1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=42,
    )

    train_idx, temp_idx = next(
        gss1.split(
            df,
            y=df["class"],
            groups=groups,
        )
    )

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    # ----------------------------------------------------------
    # Second split: temporary 30% -> 15% validation / 15% test
    # ----------------------------------------------------------

    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=42,
    )

    val_idx, test_idx = next(
        gss2.split(
            temp_df,
            y=temp_df["class"],
            groups=temp_df["sentinel_id"],
        )
    )

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    # ----------------------------------------------------------
    # Safety check: no scene leakage
    # ----------------------------------------------------------

    train_groups = set(train_df["sentinel_id"])
    val_groups = set(val_df["sentinel_id"])
    test_groups = set(test_df["sentinel_id"])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
        encoding="utf-8",
    )

    val_df.to_csv(
        OUTPUT_DIR / "val.csv",
        index=False,
        encoding="utf-8",
    )

    test_df.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
        encoding="utf-8",
    )

    all_df = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )

    all_df.to_csv(
        OUTPUT_DIR / "dataset_split_manifest.csv",
        index=False,
        encoding="utf-8",
    )

    # ----------------------------------------------------------
    # Report
    # ----------------------------------------------------------

    print()
    print("Split results:")
    print()

    for name, part in [
        ("TRAIN", train_df),
        ("VAL", val_df),
        ("TEST", test_df),
    ]:
        print(
            f"{name:5s} | "
            f"images={len(part):4d} | "
            f"scenes={part['sentinel_id'].nunique():3d} | "
            f"oil={int(part['class'].sum()):4d} | "
            f"no_oil={int((part['class'] == 0).sum()):4d}"
        )

    print()
    print("Leakage check:")
    print(
        "Train ∩ Val :",
        len(train_groups & val_groups),
    )
    print(
        "Train ∩ Test:",
        len(train_groups & test_groups),
    )
    print(
        "Val ∩ Test   :",
        len(val_groups & test_groups),
    )

    print()
    print("Files created:")
    print(OUTPUT_DIR / "train.csv")
    print(OUTPUT_DIR / "val.csv")
    print(OUTPUT_DIR / "test.csv")
    print(OUTPUT_DIR / "dataset_split_manifest.csv")


if __name__ == "__main__":
    main()
