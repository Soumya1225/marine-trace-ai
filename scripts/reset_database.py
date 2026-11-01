"""Delete all incidents, analyses and uploaded/registered files from the dashboard database.

Use it before a demo or viva to start clean. Your datasets in data/ and trained models are NOT touched.
Stop the backend first (Ctrl+C), then:
    python scripts/reset_database.py
"""
import shutil
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
db = BACKEND / "oilspill.db"
if db.exists():
    db.unlink()
    print("Deleted backend/oilspill.db")
for sub in ("results", "reports", "demo", "scenes", "ais"):
    folder = BACKEND / "uploads" / sub
    if folder.exists():
        shutil.rmtree(folder)
        print(f"Cleared backend/uploads/{sub}")
print("Done. Start the backend again; a fresh admin account will be created.")
