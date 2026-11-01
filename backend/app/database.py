from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import DATABASE_URL

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


NEW_COLUMNS = {"incidents": {"assigned_to": "INTEGER", "due_at": "DATETIME", "closed_at": "DATETIME"}}


def add_missing_columns():
    """Tiny migration: add columns introduced after the database was first created."""
    insp = inspect(engine)
    with engine.begin() as conn:
        for table, cols in NEW_COLUMNS.items():
            if table not in insp.get_table_names():
                continue
            have = {c["name"] for c in insp.get_columns(table)}
            for name, typ in cols.items():
                if name not in have:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {typ}"))
                    print(f"[db] added column {table}.{name}")
