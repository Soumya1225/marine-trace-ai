"""One-time restructure: rename the 'data' folder to 'Datasets' and update every path that points to it.

What it does (a backup of every changed file is kept in ../old_unused/before_datasets_rename/):
  1. renames  data/  ->  Datasets/   (sub-folders raw/, processed/, reference/, synthetic/ stay the same)
  2. moves empty dataset folders (nothing downloaded in them) to ../old_unused/empty_dataset_folders/
  3. updates paths in the code, scripts, docs, notebooks, .gitignore and README
  4. updates file paths saved in the database and in the saved analysis results
  5. writes Datasets/README.md (what each dataset is, where it came from, which part of the project uses it)

    python scripts/move_to_datasets.py        (run from the project folder with the backend and frontend STOPPED)
"""
import json
import os
import re
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD, NEW = ROOT / "data", ROOT / "Datasets"
BACKUP = ROOT.parent / "old_unused" / "before_datasets_rename"
EMPTY_TO = ROOT.parent / "old_unused" / "empty_dataset_folders"
SKIP_DIRS = {"venv", "node_modules", "data", "Datasets", "__pycache__", "uploads", "dist", "colab_upload",
             "backup_300cases", ".git", ".pytest_cache", ".vite"}
TEXT_EXT = {".py", ".js", ".jsx", ".bat", ".md", ".txt", ".ipynb", ".cfg", ".ini", ".toml", ".example"}
TEXT_NAMES = {".gitignore", ".env.example"}

# data/raw, data\processed, ..\data\\reference (JSON-escaped) ...  ->  Datasets/...
PATH_RE = re.compile(r"(?<![\w.\-])data(?=[/\\]+(?:raw|processed|synthetic|reference|demo)\b)")
# Path objects:  ROOT / "data"   or   / "data"  on its own line
JOIN_RE = re.compile(r'(/\s*)"data"')
# absolute paths saved in the database / results: ...oil-spill-attribution\data\...
ABS_RE = re.compile(r"(oil-spill-attribution(?:\\\\|\\|/)+)data(?=(?:\\\\|\\|/))")


def fix_text(s):
    s2 = PATH_RE.sub("Datasets", s)
    s2 = JOIN_RE.sub(r'\1"Datasets"', s2)
    return ABS_RE.sub(r"\1Datasets", s2)


def backup(path):
    dst = BACKUP / path.relative_to(ROOT)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(path, dst)


def code_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            p = Path(dirpath) / f
            if p.suffix.lower() in TEXT_EXT or f in TEXT_NAMES:
                if p.name == "move_to_datasets.py":
                    continue
                yield p


def file_count(folder):
    return sum(1 for p in folder.rglob("*") if p.is_file())


def fix_database():
    db = ROOT / "backend" / "oilspill.db"
    if not db.exists():
        return 0
    backup(db)
    con = sqlite3.connect(db)
    changed = 0
    try:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for t in tables:
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')
                    if (r[2] or "").upper() in ("", "TEXT", "JSON", "VARCHAR") or "CHAR" in (r[2] or "").upper()]
            pk = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")') if r[5]]
            key = pk[0] if pk else "rowid"
            for c in cols:
                rows = con.execute(f'SELECT "{key}", "{c}" FROM "{t}" WHERE "{c}" LIKE \'%data%\'').fetchall()
                for k, v in rows:
                    if isinstance(v, str):
                        nv = fix_text(v)
                        if nv != v:
                            con.execute(f'UPDATE "{t}" SET "{c}"=? WHERE "{key}"=?', (nv, k))
                            changed += 1
        con.commit()
    finally:
        con.close()
    return changed


def fix_results():
    n = 0
    for p in (ROOT / "backend" / "uploads").rglob("*.json"):
        s = p.read_text(encoding="utf-8", errors="ignore")
        s2 = fix_text(s)
        if s2 != s:
            backup(p)
            p.write_text(s2, encoding="utf-8")
            n += 1
    return n


README = """# Datasets

All data used by Marine Trace AI is kept in this one folder. The large files are not uploaded to GitHub
(see `.gitignore`); the download scripts in `scripts/` recreate them.

| Group | Folder | What it is | Source | Used by |
|---|---|---|---|---|
| Satellite radar (SAR) | `raw/sentinel1/` | Sentinel-1 GRD scenes of the Huntington Beach spill, 3 Oct 2021 (morning and afternoon passes) | Copernicus Data Space (`ml/preprocessing/download_sentinel1.py`) | case study analysis |
| Oil spill labels | `raw/sos/` | SOS dataset: Sentinel-1 image crops with oil masks (train / test) | Deep-SAR Oil Spill (SOS) public dataset | training and testing the detection models |
| Oil spill labels | `raw/dartis_2019/`, `processed/dartis_2019/` | DARTIS 2019 oil spill catalogue and train/val/test splits | PANGAEA (`scripts/download_dartis_*.py`) | validation |
| AIS vessel data | `raw/ais/` | US AIS ship positions, 2-3 Oct 2021, cropped to the study area | NOAA MarineCadastre (`scripts/download_ais.py`) | ship tracks, suspects, dark-vessel matching |
| Environmental | `raw/weather/`, `raw/weather_global/` | ERA5 wind (u, v) | Copernicus Climate Data Store (`scripts/download_weather.py`) | drift model, LSTM, look-alike check |
| Environmental | `raw/ocean_current/` | CMEMS surface currents (u, v) | Copernicus Marine Service | drift model, LSTM |
| Environmental | `raw/sst/`, `raw/waves/` | Sea surface temperature, wave height | Copernicus Marine Service | environmental context |
| GIS | `raw/coastline/`, `reference/` | Coastline and land polygons, sensitive coastal sites | Natural Earth (`scripts/download_coastline.py`) | land mask, landfall, impact |
| GIS | `../ml/gis/offshore_infrastructure.geojson` | Oil platforms and pipelines in San Pedro Bay (approximate positions) | NOAA IncidentNews, BOEM maps | possible non-ship source check |
| Synthetic | `synthetic/` | 1,000 simulated spill cases with known polluter (training data for the ship rankers) | `ml/attribution/synthetic_generator.py` | XGBoost, LightGBM, cross-attention ranker |
| Processed | `processed/` | Cleaned AIS tables and earlier analysis runs | generated by the pipeline | - |
"""


def main():
    if not OLD.exists():
        if NEW.exists():
            print("Already done: the Datasets folder exists.")
        else:
            print("No 'data' folder found. Run this from the project folder.")
        return
    if NEW.exists():
        print("Both 'data' and 'Datasets' exist. Stop: merge them by hand first.")
        return
    try:
        OLD.rename(NEW)
    except OSError as e:
        print("Could not rename the folder (a program is using it). Stop the backend/frontend, close File Explorer "
              "windows inside 'data', then run again.\n", e)
        sys.exit(1)
    print("1. Renamed data -> Datasets")

    moved = []
    for rel in ["raw/krestenitis", "raw/lados", "raw/MADOS", "processed/mados_optical"]:
        p = NEW / rel
        if p.exists() and file_count(p) == 0:
            dst = EMPTY_TO / rel.replace("/", "_")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dst))
            moved.append(rel)
    print(f"2. Moved {len(moved)} empty dataset folders out: {', '.join(moved) or 'none'}")

    changed = []
    for p in code_files():
        s = p.read_text(encoding="utf-8", errors="ignore")
        s2 = fix_text(s)
        if s2 != s:
            backup(p)
            p.write_text(s2, encoding="utf-8")
            changed.append(str(p.relative_to(ROOT)))
    print(f"3. Updated paths in {len(changed)} files:")
    for c in changed:
        print("     ", c)

    print(f"4. Updated {fix_database()} database values and {fix_results()} saved result files")
    (NEW / "README.md").write_text(README, encoding="utf-8")
    print("5. Wrote Datasets/README.md")
    print(f"Backup of every changed file: {BACKUP}  ({datetime.now():%Y-%m-%d %H:%M})")


if __name__ == "__main__":
    main()
