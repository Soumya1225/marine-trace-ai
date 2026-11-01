from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def now():
    return datetime.utcnow()


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    role: Mapped[str] = mapped_column(String(20), default="analyst")      # admin | analyst | viewer
    password_hash: Mapped[str] = mapped_column(String(300))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class Scene(Base):
    __tablename__ = "scenes"
    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(300))
    path: Mapped[str] = mapped_column(String(600))
    acquisition_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    bbox: Mapped[list | None] = mapped_column(JSON, nullable=True)
    size_mb: Mapped[float] = mapped_column(Float, default=0)
    uploaded_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class AISFile(Base):
    __tablename__ = "ais_files"
    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(300))
    path: Mapped[str] = mapped_column(String(600))
    size_mb: Mapped[float] = mapped_column(Float, default=0)
    uploaded_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"
    id: Mapped[int] = mapped_column(primary_key=True)
    scene_id: Mapped[int] = mapped_column(ForeignKey("scenes.id"))
    ais_ids: Mapped[list] = mapped_column(JSON, default=list)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    state: Mapped[str] = mapped_column(String(20), default="queued")       # queued | running | done | failed
    progress: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str] = mapped_column(String(300), default="Waiting to start")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    scene: Mapped["Scene"] = relationship()


class Incident(Base):
    __tablename__ = "incidents"
    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("analysis_jobs.id"))
    scene_id: Mapped[int] = mapped_column(ForeignKey("scenes.id"))
    spill_no: Mapped[int] = mapped_column(Integer, default=1)
    title: Mapped[str] = mapped_column(String(300))
    status: Mapped[str] = mapped_column(String(20), default="detected")    # detected | under_review | confirmed | closed | rejected
    priority: Mapped[str] = mapped_column(String(20), default="Medium")
    priority_score: Mapped[float] = mapped_column(Float, default=0)
    area_km2: Mapped[float] = mapped_column(Float, default=0)
    centroid_lat: Mapped[float] = mapped_column(Float)
    centroid_lon: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    detected_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    result_dir: Mapped[str] = mapped_column(String(600))
    data: Mapped[dict] = mapped_column(JSON, default=dict)                  # spill, forecast, impact, decision, backtrack
    assigned_to: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    due_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    assignee: Mapped[Optional["User"]] = relationship(foreign_keys=[assigned_to])
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now, onupdate=now)
    suspects: Mapped[list["Suspect"]] = relationship(back_populates="incident", cascade="all, delete-orphan",
                                                     order_by="Suspect.rank")
    notes: Mapped[list["Note"]] = relationship(back_populates="incident", cascade="all, delete-orphan",
                                               order_by="Note.created_at")


class Suspect(Base):
    __tablename__ = "suspects"
    id: Mapped[int] = mapped_column(primary_key=True)
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id"))
    mmsi: Mapped[int] = mapped_column(Integer, index=True)
    vessel_name: Mapped[str] = mapped_column(String(200))
    vessel_type: Mapped[str] = mapped_column(String(60))
    rank: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)
    rule_score: Mapped[float] = mapped_column(Float)
    ml_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    level: Mapped[str] = mapped_column(String(10))
    features: Mapped[dict] = mapped_column(JSON, default=dict)
    evidence: Mapped[list] = mapped_column(JSON, default=list)
    track: Mapped[list] = mapped_column(JSON, default=list)
    review_status: Mapped[str] = mapped_column(String(20), default="pending")   # pending | confirmed | dismissed
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    incident: Mapped["Incident"] = relationship(back_populates="suspects")


class Vessel(Base):
    __tablename__ = "vessels"
    mmsi: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    imo: Mapped[str | None] = mapped_column(String(30), nullable=True)
    vessel_type: Mapped[str] = mapped_column(String(60), default="Unknown")
    length_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Note(Base):
    __tablename__ = "notes"
    id: Mapped[int] = mapped_column(primary_key=True)
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    incident: Mapped["Incident"] = relationship(back_populates="notes")
    user: Mapped["User"] = relationship()


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(primary_key=True)
    incident_id: Mapped[int | None] = mapped_column(ForeignKey("incidents.id"), nullable=True)
    level: Mapped[str] = mapped_column(String(20))
    message: Mapped[str] = mapped_column(Text)
    read: Mapped[bool] = mapped_column(Boolean, default=False)
    emailed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class ActivityLog(Base):
    __tablename__ = "activity_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(60))
    detail: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    user: Mapped["User"] = relationship()


class WatchZone(Base):
    """An area of sea that is monitored (used to flag spills, filter the map and, later, for automatic image checks)."""
    __tablename__ = "watch_zones"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    min_lon: Mapped[float] = mapped_column(Float)
    min_lat: Mapped[float] = mapped_column(Float)
    max_lon: Mapped[float] = mapped_column(Float)
    max_lat: Mapped[float] = mapped_column(Float)
    notes: Mapped[str] = mapped_column(Text, default="")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    auto_check: Mapped[bool] = mapped_column(Boolean, default=False)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class SatellitePass(Base):
    """A Sentinel-1 image found by the automatic feed over a watch zone."""
    __tablename__ = "satellite_passes"
    id: Mapped[int] = mapped_column(primary_key=True)
    zone_id: Mapped[int] = mapped_column(ForeignKey("watch_zones.id"))
    product_id: Mapped[str] = mapped_column(String(80))
    name: Mapped[str] = mapped_column(String(300))
    acquired_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    size_mb: Mapped[float] = mapped_column(Float, default=0)
    footprint: Mapped[list | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="new")   # new | downloading | analysing | done | failed | skipped
    message: Mapped[str] = mapped_column(String(400), default="")
    scene_id: Mapped[int | None] = mapped_column(ForeignKey("scenes.id"), nullable=True)
    job_id: Mapped[int | None] = mapped_column(ForeignKey("analysis_jobs.id"), nullable=True)
    found_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    zone: Mapped["WatchZone"] = relationship()
