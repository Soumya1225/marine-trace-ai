"""Pack what Google Colab needs for training into two zip files in colab_upload/.

    python scripts/make_colab_bundle.py
Creates:
  colab_upload/project_code.zip   - the ml/ package (small)
  colab_upload/sos_sentinel.zip   - SOS Sentinel-1 images and masks (train + test)
Upload both files to Google Drive, folder: MyDrive/MarineTraceAI/
"""
import zipfile
from pathlib import Path

OUT = Path("colab_upload")


def add_tree(z, folder, skip=("__pycache__",)):
    count = 0
    for p in Path(folder).rglob("*"):
        if p.is_file() and not any(s in p.parts for s in skip):
            z.write(p, p.as_posix())
            count += 1
    return count


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT / "project_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        n = add_tree(z, "ml")
    print(f"project_code.zip: {n} files")

    with zipfile.ZipFile(OUT / "sos_sentinel.zip", "w", zipfile.ZIP_STORED) as z:
        n = add_tree(z, "Datasets/raw/sos/train/sentinel") + add_tree(z, "Datasets/raw/sos/test/sentinel")
    size = (OUT / "sos_sentinel.zip").stat().st_size / 1e6
    print(f"sos_sentinel.zip: {n} files, {size:.0f} MB")
    print(f"Done. Upload both files from {OUT.resolve()} to Google Drive -> MyDrive/MarineTraceAI/")
