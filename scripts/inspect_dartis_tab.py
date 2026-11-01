from pathlib import Path
import pandas as pd

path = Path(r".\Datasets\raw\dartis_2019\DARTIS_2019.tab")

lines = path.read_text(
    encoding="utf-8",
    errors="replace"
).splitlines()

header_index = None

for i, line in enumerate(lines):
    if line.startswith("Image set (subset;"):
        header_index = i
        break

if header_index is None:
    raise RuntimeError(
        "Could not find the DARTIS table header."
    )

print("Total lines:", len(lines))
print("Header line:", header_index + 1)
print("Data starts:", header_index + 2)

df = pd.read_csv(
    path,
    sep="\t",
    skiprows=header_index,
    engine="python",
)

print()
print("Rows:", len(df))
print("Columns:", len(df.columns))
print()

print("Columns:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i:02d}. {column}")

print()
print("First 3 records:")
print(df.head(3).to_string())

# Clean column names for use in MarineTraceAI
clean_columns = []

for column in df.columns:
    name = str(column).strip()

    if "Image set" in name:
        name = "image_set"
    elif "IMAGE" in name:
        name = "jpg_file"
    elif "Binary" in name:
        name = "xml_file"
    elif name.startswith("ID (tag)"):
        name = "tag"
    elif name.startswith("ID (patch_name)"):
        name = "patch_name"
    elif "Date/Time (start_time)" in name:
        name = "start_time"
    elif "Date/Time (end_time)" in name:
        name = "end_time"
    elif "ID (Sentinel_ID)" in name:
        name = "sentinel_id"

    clean_columns.append(name)

df.columns = clean_columns

output = Path(
    r".\Datasets\processed\dartis_2019"
)

output.mkdir(
    parents=True,
    exist_ok=True
)

clean_file = output / "dartis_2019_metadata.csv"

df.to_csv(
    clean_file,
    index=False,
    encoding="utf-8"
)

print()
print("Clean metadata saved to:")
print(clean_file)

print()
print("Image subsets:")
if "image_set" in df.columns:
    print(
        df["image_set"]
        .astype(str)
        .str.strip()
        .value_counts()
        .to_string()
    )

print()
print("XML annotations available:",
      "xml_file" in df.columns)

print("JPEG filenames available:",
      "jpg_file" in df.columns)
