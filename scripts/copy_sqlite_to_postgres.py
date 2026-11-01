"""Copy all Marine Trace AI data from the old SQLite file to PostgreSQL.

Run from the project folder:  python scripts/copy_sqlite_to_postgres.py
Reads DATABASE_URL from the root .env file. The SQLite file is only read, never changed.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from dotenv import dotenv_values  # noqa: E402
from sqlalchemy import create_engine, inspect, select, text  # noqa: E402

target_url = dotenv_values(ROOT / ".env").get("DATABASE_URL") or os.getenv("DATABASE_URL")
sqlite_file = BACKEND / "oilspill.db"
if not target_url or target_url.startswith("sqlite"):
    sys.exit("DATABASE_URL in .env is not set to PostgreSQL. Stopping.")
if not sqlite_file.exists():
    sys.exit(f"SQLite file not found: {sqlite_file}")

os.environ["DATABASE_URL"] = target_url
from app.database import Base, engine as target  # noqa: E402
import app.models  # noqa: E402,F401  (registers all tables)

source = create_engine(f"sqlite:///{sqlite_file.as_posix()}")
Base.metadata.create_all(bind=target)
is_pg = target.dialect.name == "postgresql"
insp = inspect(source)
src_tables = set(insp.get_table_names())

with source.connect() as src, target.begin() as dst:
    if is_pg:
        dst.execute(text("SET session_replication_role = replica"))
    for t in reversed(Base.metadata.sorted_tables):
        dst.execute(t.delete())
    total = 0
    for t in Base.metadata.sorted_tables:
        if t.name not in src_tables:
            print(f"  {t.name:<22} not in SQLite, skipped")
            continue
        have = {c["name"] for c in insp.get_columns(t.name)}
        cols = [c for c in t.columns if c.name in have]
        rows = [dict(r._mapping) for r in src.execute(select(*cols))]
        if rows:
            dst.execute(t.insert(), rows)
        total += len(rows)
        print(f"  {t.name:<22} {len(rows):>6} rows")
        if is_pg and "id" in t.c and t.c.id.primary_key:
            dst.execute(text(
                f"SELECT setval(pg_get_serial_sequence('\"{t.name}\"', 'id'), "
                f"COALESCE((SELECT MAX(id) FROM \"{t.name}\"), 0) + 1, false)"))
    if is_pg:
        dst.execute(text("SET session_replication_role = origin"))

print(f"Done: {total} rows copied to {target.dialect.name}.")
