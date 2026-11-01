"""Request bodies (responses are plain dicts built in the routers)."""
from datetime import datetime

from pydantic import BaseModel, Field


class LoginIn(BaseModel):
    email: str
    password: str


class UserCreate(BaseModel):
    name: str
    email: str
    password: str = Field(min_length=6)
    role: str = "analyst"


class UserUpdate(BaseModel):
    name: str | None = None
    role: str | None = None
    active: bool | None = None
    password: str | None = None


class RegisterPath(BaseModel):
    path: str
    acquisition_time: datetime | None = None
    bbox: list[float] | None = None


class AnalyzeIn(BaseModel):
    scene_id: int
    ais_ids: list[int] = []
    bbox: list[float] | None = None              # min_lon, min_lat, max_lon, max_lat
    acquisition_time: datetime | None = None
    wind_u: float | None = None
    wind_v: float | None = None
    current_u: float | None = None
    current_v: float | None = None


class StatusUpdate(BaseModel):
    status: str
    note: str | None = None


class NoteIn(BaseModel):
    text: str = Field(min_length=1)


class SuspectReview(BaseModel):
    review_status: str                            # confirmed | dismissed | pending
    review_note: str | None = None
