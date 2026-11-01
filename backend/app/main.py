"""FastAPI application entry point.

Run from the backend folder:
    uvicorn app.main:app --reload
API docs: http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import config
from app.auth import hash_password
from app.database import Base, SessionLocal, add_missing_columns, engine
from app.models import User
from app.security import HEADERS as SECURITY_HEADERS
from app.routers import auth, cases, feed, feedback, incidents, live, quickcheck, scenes, stats, system, zones

app = FastAPI(title="Oil Spill Detection & Vessel Attribution API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    for k, v in SECURITY_HEADERS.items():
        response.headers.setdefault(k, v)
    return response


for r in (auth.router, scenes.router, incidents.router, stats.router, quickcheck.router, feedback.router, system.router, cases.router, zones.router, feed.router, live.router):
    app.include_router(r)

app.mount("/files", StaticFiles(directory=config.STORAGE_DIR / "results"), name="files")
figures = config.PROJECT_ROOT / "reports" / "figures"
figures.mkdir(parents=True, exist_ok=True)
app.mount("/figures", StaticFiles(directory=figures), name="figures")


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    add_missing_columns()
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            db.add(User(name="Administrator", email=config.ADMIN_EMAIL.lower(), role="admin",
                        password_hash=hash_password(config.ADMIN_PASSWORD)))
            db.commit()
            print(f"Created admin user {config.ADMIN_EMAIL} / {config.ADMIN_PASSWORD} (change this password)")
    finally:
        db.close()
    # Load the AI detector once here (main thread). Loading PyTorch for the first time inside
    # a background job can fail on Windows, which silently falls back to the baseline detector.
    from app.services.analysis_service import get_detector
    det = get_detector()
    print(f"Detector loaded: {det.mode} ({det.arch})")
    if config.SECRET_KEY.startswith("change-this"):
        print("WARNING: SECRET_KEY is the default value. Set a long random SECRET_KEY in .env before sharing the app.")
    from app.services.satellite_feed import start_scheduler
    start_scheduler()
    from app.services.ais_live import LIVE
    LIVE.start()


@app.get("/api/health")
def health():
    return {"status": "ok"}
